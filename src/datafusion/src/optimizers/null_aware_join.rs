//! Keeps dynamic filters away from null-aware anti joins.
//!
//! `x NOT IN (<subquery>)` plans as a null-aware `LeftAnti` hash join, and its
//! answer turns on what the probe side *contains*, not just on which probe rows
//! match:
//!
//! - a NULL probe key annihilates the output (`x NOT IN (.., NULL, ..)` is
//!   never true), and
//! - an empty probe side passes every build row through, NULL keys included.
//!
//! A dynamic filter destroys both signals. `HashJoinExec` publishes one built
//! from its build-side keys (`key >= min AND key <= max AND key IN (..)`) and
//! pushes it into the probe-side scan; `SortExec`'s TopK and the aggregate
//! rules publish their own. NULL satisfies no comparison, so probe NULLs are
//! pruned before the join ever sees them, and a probe side holding no matching
//! keys prunes away to nothing and reads as empty. Either way the join answers
//! as if the probe side had no NULLs, and `NOT IN` returns rows that SQL
//! tri-state semantics require it to drop.
//!
//! Removing the filter from the null-aware join alone is not enough: a join
//! *above* it pushes its own filter straight through into the same probe
//! subtree, because `HashJoinExec` reports both sides of a `LeftAnti` as
//! preserved for pushdown ("join key filters can be safely pushed down into
//! the other side" — true for a plain anti join, false for a null-aware one).
//! So the guard works at plan scope instead: when the plan contains a
//! null-aware join at all, every rule runs with dynamic filter pushdown turned
//! off, and no dynamic filter is created anywhere in that plan.
//!
//! Only static predicate pushdown is left alone, which is sound here — the
//! probe side *is* the statically filtered subquery, so pruning it by its own
//! predicate cannot change the answer. The cost is that a query containing a
//! `NOT IN` gives up dynamic-filter pruning on its other joins too; that is the
//! granularity the config exposes, and correctness wins.
//!
//! Upstream guards this only when the *build* key is nullable, which misses the
//! pruned-probe-NULL case entirely. See
//! <https://github.com/hotdata-dev/liquid-cache/issues/16>.

use std::sync::Arc;

use datafusion::{
    common::tree_node::{TreeNode, TreeNodeRecursion},
    config::ConfigOptions,
    error::Result,
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::{ExecutionPlan, joins::HashJoinExec},
};

/// Wraps a physical optimizer rule so it never gets to attach a dynamic filter
/// to a plan that contains a null-aware anti join. See the module docs.
///
/// Wrap every rule in the list — dynamic filters are created inside the
/// filter-pushdown rules today, but nothing pins them there, and the wrapper is
/// inert for a plan with no null-aware join.
#[derive(Debug)]
pub struct NoDynamicFiltersForNullAwareJoins {
    inner: Arc<dyn PhysicalOptimizerRule + Send + Sync>,
}

impl NoDynamicFiltersForNullAwareJoins {
    /// Wrap `inner`.
    pub fn new(inner: Arc<dyn PhysicalOptimizerRule + Send + Sync>) -> Self {
        Self { inner }
    }

    /// Wrap every rule of DataFusion's default physical optimizer, ready to
    /// hand to `SessionStateBuilder::with_physical_optimizer_rules`.
    pub fn wrap_default_rules() -> Vec<Arc<dyn PhysicalOptimizerRule + Send + Sync>> {
        datafusion::physical_optimizer::optimizer::PhysicalOptimizer::new()
            .rules
            .into_iter()
            .map(|rule| Arc::new(Self::new(rule)) as Arc<dyn PhysicalOptimizerRule + Send + Sync>)
            .collect()
    }
}

/// Whether `plan` contains a hash join asking for null-aware semantics.
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

impl PhysicalOptimizerRule for NoDynamicFiltersForNullAwareJoins {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !has_null_aware_join(&plan) {
            return self.inner.optimize(plan, config);
        }
        let mut config = config.clone();
        let optimizer = &mut config.optimizer;
        optimizer.enable_dynamic_filter_pushdown = false;
        optimizer.enable_join_dynamic_filter_pushdown = false;
        optimizer.enable_topk_dynamic_filter_pushdown = false;
        optimizer.enable_aggregate_dynamic_filter_pushdown = false;
        self.inner.optimize(plan, &config)
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn schema_check(&self) -> bool {
        self.inner.schema_check()
    }
}
