//! Keeps join dynamic filters off the probe side of null-aware anti joins.
//!
//! `x NOT IN (<subquery>)` plans as a null-aware `LeftAnti` hash join, and its
//! answer turns on what the probe side *contains*, not just on which probe rows
//! match:
//!
//! - a NULL probe key annihilates the output (`x NOT IN (.., NULL, ..)` is
//!   never true), and
//! - an empty probe side passes every build row through, NULL keys included.
//!
//! `HashJoinExec` publishes a filter built from its build-side keys
//! (`key >= min AND key <= max AND key IN (..)`) and pushes it into its probe
//! side. That is sound for a plain anti join — a probe row that cannot match
//! cannot change which build rows are unmatched — but it destroys both signals
//! above: NULL satisfies no comparison, so probe NULLs are pruned before the
//! join sees them, and a probe side holding no matching keys prunes away to
//! nothing and reads as empty. Either way the join answers as if the probe side
//! had no NULLs.
//!
//! A join *above* the null-aware one does the same damage with its own filter,
//! because `HashJoinExec` reports both sides of a `LeftAnti` as preserved for
//! pushdown, so a parent's filter propagates into the anti join's probe side
//! too. So this rule clears the dynamic filter of any hash join that is itself
//! null-aware, or whose probe subtree holds a null-aware join.
//!
//! A join holding the null-aware join on its *build* side is left alone: its
//! filter goes only to its own probe side and can never reach the anti join, so
//! that pruning is sound and worth keeping. This matters — it is the difference
//! between pruning an unrelated 400k-row scan to one row and reading all of it.
//!
//! TopK and aggregate filters need the blunter treatment in
//! [`NullAwareValueFilterGuard`], because they are tightened from the value
//! stream rather than published at a fixed point in the plan. It is tempting to
//! argue they are safe — a `LeftAnti` emits nothing until its probe side is
//! fully drained, so it cannot tighten a threshold against its own unread probe
//! rows — but that only holds while the anti join is the *sole* producer
//! feeding the node. Put a second producer under the same TopK or aggregate
//! (`.. UNION ALL <NOT IN ..>`) and the sibling drains first, tightening the
//! shared filter that has already been pushed into the still-unread probe scan.
//! `min()`/`max()` filters and nulls-last TopK filters drop NULLs outright, so
//! the deciding NULL disappears.
//!
//! What both guards deliberately leave alone:
//!
//! - **Static predicate pushdown.** Not because it is harmless in principle —
//!   the physical `lr_is_preserved` does route a static parent predicate into a
//!   `LeftAnti`'s probe side — but because the logical optimizer already copies
//!   any join-key predicate onto the subquery side before a physical plan
//!   exists (see below), so nothing is left here to protect. If upstream fixes
//!   that without fixing physical `lr_is_preserved`, revisit this exemption.
//!
//! Two holes remain that no physical rule can reach, both upstream:
//!
//! - `infer_join_predicates` copies *any* predicate on the join key onto the
//!   subquery side, `IS NOT NULL` included, deleting the deciding NULL. So
//!   `a NOT IN (..) AND <anything on a>` is still wrong — wrong in vanilla
//!   DataFusion over a `MemTable` with every dynamic filter off.
//! - With `prefer_hash_join = false` the planner emits a `SortMergeJoinExec`
//!   and drops `null_aware` outright. That leaves no `HashJoinExec` to guard,
//!   so *both* guards go inert — the value guard keys off the same detection as
//!   the join one, and is not independent of the hash-join path.
//!
//! Upstream guards the join's own filter only when the *build* key is nullable,
//! which misses the pruned-probe-NULL case. See
//! <https://github.com/hotdata-dev/liquid-cache/issues/16>.

use std::sync::Arc;

use datafusion::{
    common::tree_node::{Transformed, TransformedResult, TreeNode, TreeNodeRecursion},
    config::ConfigOptions,
    error::Result,
    physical_optimizer::{
        PhysicalOptimizerRule,
        optimizer::{PhysicalOptimizer, PhysicalOptimizerContext},
    },
    physical_plan::{ExecutionPlan, joins::HashJoinExec, operator_statistics::StatisticsRegistry},
};

/// Physical optimizer rule that clears the dynamic filter from hash joins whose
/// filter could reach a null-aware anti join's probe side. See the module docs.
///
/// Pairs with [`NullAwareValueFilterGuard`], which handles the TopK and
/// aggregate filters this rule does not touch.
///
/// Must run after the filter-pushdown rules that attach the dynamic filter, so
/// it belongs at the end of the physical optimizer list.
#[derive(Debug, Default)]
pub struct NullAwareJoinFilterGuard;

impl NullAwareJoinFilterGuard {
    /// Create the rule.
    pub fn new() -> Self {
        Self
    }
}

/// Whether `plan` holds a hash join asking for null-aware semantics.
fn has_null_aware_join(plan: &Arc<dyn ExecutionPlan>) -> bool {
    let mut found = false;
    // `apply` only fails if the closure does, and this one cannot.
    let _ = plan.apply(|node| {
        Ok(match node.downcast_ref::<HashJoinExec>() {
            Some(join) if join.null_aware => {
                found = true;
                TreeNodeRecursion::Stop
            }
            _ => TreeNodeRecursion::Continue,
        })
    });
    found
}

impl PhysicalOptimizerRule for NullAwareJoinFilterGuard {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        plan.transform_up(|node| {
            let Some(join) = node.downcast_ref::<HashJoinExec>() else {
                return Ok(Transformed::no(node));
            };
            if join.dynamic_filter_expr().is_none() {
                return Ok(Transformed::no(node));
            }
            // The join's own filter lands on its probe side, and a parent's
            // filter propagates down into it, so a null-aware join anywhere in
            // that subtree is equally exposed.
            if !join.null_aware && !has_null_aware_join(join.right()) {
                return Ok(Transformed::no(node));
            }
            // `reset_state` clears the dynamic filter along with the build-side
            // future and metrics, neither of which holds anything at plan time.
            // The filter expression already attached to the probe scan is then
            // never refreshed off its initial `true`, so it prunes nothing.
            Ok(Transformed::yes(join.builder().reset_state().build_exec()?))
        })
        .data()
    }

    fn name(&self) -> &str {
        "NullAwareJoinFilterGuard"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

/// A [`PhysicalOptimizerContext`] serving an overridden config while passing
/// everything else — notably the statistics registry — through untouched.
struct OverriddenConfig<'a> {
    config: ConfigOptions,
    inner: &'a dyn PhysicalOptimizerContext,
}

impl PhysicalOptimizerContext for OverriddenConfig<'_> {
    fn config_options(&self) -> &ConfigOptions {
        &self.config
    }

    fn statistics_registry(&self) -> Option<&StatisticsRegistry> {
        self.inner.statistics_registry()
    }
}

/// Wraps a physical optimizer rule so it cannot create a TopK or aggregate
/// dynamic filter while the plan holds a null-aware anti join. See the module
/// docs for why those two cannot be gated per-node the way join filters are.
///
/// Unlike the join filters, these are suppressed for the whole plan, so a
/// `NOT IN` costs the query its TopK and aggregate pruning. The join filters —
/// the expensive ones — keep their per-node treatment in
/// [`NullAwareJoinFilterGuard`].
///
/// Wrap every rule rather than only the filter-pushdown ones: matching rules by
/// name would silently stop guarding if upstream renamed one, and the cost of a
/// short-circuiting walk per rule is immeasurable.
#[derive(Debug)]
pub struct NullAwareValueFilterGuard {
    inner: Arc<dyn PhysicalOptimizerRule + Send + Sync>,
}

impl NullAwareValueFilterGuard {
    /// Wrap `inner`.
    pub fn new(inner: Arc<dyn PhysicalOptimizerRule + Send + Sync>) -> Self {
        Self { inner }
    }

    /// Wrap every rule of DataFusion's default physical optimizer, ready for
    /// `SessionStateBuilder::with_physical_optimizer_rules`.
    pub fn wrap_default_rules() -> Vec<Arc<dyn PhysicalOptimizerRule + Send + Sync>> {
        PhysicalOptimizer::new()
            .rules
            .into_iter()
            .map(|rule| Arc::new(Self::new(rule)) as Arc<dyn PhysicalOptimizerRule + Send + Sync>)
            .collect()
    }

    /// The config to run `inner` under, or `None` to pass the caller's through.
    fn override_for(
        &self,
        plan: &Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> Option<ConfigOptions> {
        let optimizer = &config.optimizer;
        if !optimizer.enable_topk_dynamic_filter_pushdown
            && !optimizer.enable_aggregate_dynamic_filter_pushdown
        {
            return None;
        }
        if !has_null_aware_join(plan) {
            return None;
        }
        let mut config = config.clone();
        let optimizer = &mut config.optimizer;
        optimizer.enable_topk_dynamic_filter_pushdown = false;
        optimizer.enable_aggregate_dynamic_filter_pushdown = false;
        Some(config)
    }
}

impl PhysicalOptimizerRule for NullAwareValueFilterGuard {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        match self.override_for(&plan, config) {
            Some(config) => self.inner.optimize(plan, &config),
            None => self.inner.optimize(plan, config),
        }
    }

    /// Forwarded, not left to the trait default: the planner calls this one, and
    /// the default would drop `inner`'s own override — and with it the
    /// statistics registry that drives join-side selection.
    fn optimize_with_context(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        context: &dyn PhysicalOptimizerContext,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        match self.override_for(&plan, context.config_options()) {
            Some(config) => self.inner.optimize_with_context(
                plan,
                &OverriddenConfig {
                    config,
                    inner: context,
                },
            ),
            None => self.inner.optimize_with_context(plan, context),
        }
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn schema_check(&self) -> bool {
        self.inner.schema_check()
    }
}

#[cfg(test)]
mod tests {
    use datafusion::config::ConfigOptions;

    /// The guards cover the dynamic-filter publishers by name: the join filter
    /// per-node, the TopK and aggregate filters plan-wide. A fifth publisher
    /// behind a new `enable_*dynamic_filter*` flag would silently slip past
    /// both, so pin the set rather than discover it from a wrong answer.
    #[test]
    fn dynamic_filter_flag_set_has_not_grown() {
        let mut keys: Vec<String> = ConfigOptions::default()
            .entries()
            .into_iter()
            .map(|entry| entry.key)
            .filter(|key| key.contains("dynamic_filter"))
            .collect();
        keys.sort();
        assert_eq!(
            keys,
            [
                "datafusion.optimizer.enable_aggregate_dynamic_filter_pushdown",
                "datafusion.optimizer.enable_dynamic_filter_pushdown",
                "datafusion.optimizer.enable_join_dynamic_filter_pushdown",
                "datafusion.optimizer.enable_topk_dynamic_filter_pushdown",
            ],
            "a new dynamic-filter flag means a new publisher to guard; see the module docs"
        );
    }
}
