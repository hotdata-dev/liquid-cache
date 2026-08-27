//! Strips the build-side dynamic filter from null-aware anti joins.
//!
//! `x NOT IN (<subquery>)` plans as a null-aware `LeftAnti` hash join, whose
//! semantics are driven by what the probe side *contains*, not just by which
//! probe rows match:
//!
//! - a NULL probe key annihilates the output (`x NOT IN (.., NULL, ..)` is
//!   never true), and
//! - an empty probe side passes every build row through, NULL keys included.
//!
//! `HashJoinExec` also publishes a dynamic filter derived from the build-side
//! keys (`key >= min AND key <= max AND key IN (..)`) and pushes it into the
//! probe-side scan. That filter is sound for a plain anti join — a probe row
//! that cannot match cannot change which build rows are unmatched — but it
//! destroys both signals above: NULL satisfies no comparison, so probe NULLs
//! are pruned before the join ever sees them, and a probe side with no
//! matching keys prunes down to nothing and reads as empty.
//!
//! Either way the join answers as if the probe side had no NULLs, so
//! `SELECT .. WHERE a NOT IN (SELECT b FROM ..)` returns the rows that SQL
//! tri-state semantics require it to filter out. Upstream guards this only when
//! the *build* key is nullable, which misses the pruned-probe-NULL case, so the
//! rule below drops the dynamic filter from every null-aware join.
//!
//! The join keeps working without it: the filter is an optional pruning
//! accelerator, and with it detached the shared expression is never refreshed
//! from the build side and stays `true`. Only null-aware joins are touched, and
//! a null-aware join always has a nullable join key (otherwise the planner
//! would not have asked for null-aware semantics), so there is no case where
//! this gives up pruning it could soundly have kept.
//!
//! See <https://github.com/hotdata-dev/liquid-cache/issues/16>.

use std::sync::Arc;

use datafusion::{
    common::tree_node::{Transformed, TransformedResult, TreeNode},
    config::ConfigOptions,
    error::Result,
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::{ExecutionPlan, joins::HashJoinExec},
};

/// Physical optimizer rule that removes the build-side dynamic filter from
/// null-aware anti joins, which cannot read their probe side correctly through
/// it. See the module docs.
///
/// Must run after the filter-pushdown rules that attach the dynamic filter,
/// i.e. it belongs at the end of the physical optimizer list.
#[derive(Debug, Default)]
pub struct NullAwareJoinDynamicFilterGuard;

impl NullAwareJoinDynamicFilterGuard {
    /// Create the rule.
    pub fn new() -> Self {
        Self
    }
}

impl PhysicalOptimizerRule for NullAwareJoinDynamicFilterGuard {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        plan.transform_up(|node| {
            let Some(join) = node.downcast_ref::<HashJoinExec>() else {
                return Ok(Transformed::no(node));
            };
            if !join.null_aware || join.dynamic_filter_expr().is_none() {
                return Ok(Transformed::no(node));
            }
            // `reset_state` clears the dynamic filter along with the build-side
            // future and metrics, none of which carry anything yet at plan time.
            Ok(Transformed::yes(join.builder().reset_state().build_exec()?))
        })
        .data()
    }

    fn name(&self) -> &str {
        "NullAwareJoinDynamicFilterGuard"
    }

    fn schema_check(&self) -> bool {
        true
    }
}
