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
//! What it deliberately leaves alone:
//!
//! - **A join holding the null-aware join on its *build* side.** Its filter
//!   goes only to its own probe side and can never reach the anti join, so that
//!   pruning is sound and worth keeping.
//! - **TopK and aggregate dynamic filters.** Those tighten from the value
//!   stream *above* the join, and a `LeftAnti` emits nothing until its probe
//!   side is fully drained, so they cannot prune a probe row before the join
//!   has counted it. Only the join's own filter comes from the build side,
//!   which is collected first.
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
//!   and drops `null_aware` outright, leaving no `HashJoinExec` to guard.
//!
//! Upstream guards the join's own filter only when the *build* key is nullable,
//! which misses the pruned-probe-NULL case. See
//! <https://github.com/hotdata-dev/liquid-cache/issues/16>.

use std::sync::Arc;

use datafusion::{
    common::tree_node::{Transformed, TransformedResult, TreeNode, TreeNodeRecursion},
    config::ConfigOptions,
    error::Result,
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::{ExecutionPlan, joins::HashJoinExec},
};

/// Physical optimizer rule that clears the dynamic filter from hash joins whose
/// filter could reach a null-aware anti join's probe side. See the module docs.
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
