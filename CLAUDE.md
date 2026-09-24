# Working in this fork

`hotdata-dev/liquid-cache` is a fork of `datafusion-contrib/liquid-cache`.
`main` is upstream's `0033b15` with our patches on top — upstream's history is
fully contained, so `git merge-base main <upstream commit>` resolves.

`AGENTS.md` and `README.md` are upstream's and describe the project itself.
This file is ours and describes only what differs here. Nothing in this repo
should edit an upstream-owned file to record a fork convention: that conflicts
on every sync. Add a file upstream does not have instead.

## Branches

- Work off `main`. `git fetch fork` first — a local `main` goes stale with no
  signal, and branching off a stale one silently drops everything merged since.
- **Upstream PRs branch from upstream, not from `main`**, and are named
  `upstream/<topic>`. A branch cut from `main` carries our whole patch stack
  into the PR diff.

  ```
  git fetch https://github.com/datafusion-contrib/liquid-cache main
  git checkout -b upstream/<topic> FETCH_HEAD
  ```

  `.github/workflows/upstream-branch-guard.yml` fails any `upstream/**` branch
  that descends from `main`, on push, before a PR exists.

- Before raising an upstream PR, reproduce the bug **on upstream's tree** —
  apply the test alone, watch it fail, then apply the fix. A fix whose code
  still exists upstream is not evidence the bug does; that mistake has cost us
  a withdrawn PR. Several of our patches repair machinery upstream does not
  have (`DiskResidue`, `reclaim_orphaned_disk`, `settle`) and are not
  upstreamable at all.

## Building and testing

- **`cargo +1.96.0`.** The dependency tree needs 1.95+ (`vortex-*`, `sysinfo`)
  and DataFusion 55 needs 1.94. A bare `cargo` on an older default fails
  resolution with a wall of `requires rustc 1.9x` lines.
- **Shuttle tests need a filter**: `cargo +1.96.0 test -p liquid-cache
  --features shuttle --lib shuttle_`. The feature swaps `crate::sync` to
  shuttle primitives for the whole test build, so running it unfiltered fails
  ~49 unrelated tests with "Are you accessing a Shuttle primitive outside of a
  Shuttle test?". That is by design, not a regression.
- **`dev-tools` needs `dev/dev-tools/assets/tailwind.css`**, which CI generates
  and the repo does not carry. To run its tests locally, create a placeholder
  and delete it before committing. Do not habitually pass
  `--exclude dev-tools`: it hides real failures, including trace-snapshot
  breakage from new cache events.
- After a structural edit, compare the **test inventory**, not just the pass
  count. A `#[cfg(feature = "shuttle")]` test can be deleted without moving the
  default-build total at all, and CI stays green because a missing test is not
  a failing one.

## Known open issue

The disk reclaim path has an unclosed race. A store key is
`(entry id, identity)` and identities are reused — the file-id pool hands a
re-opened path its previous record. `reclaim_orphaned_disk` consults the index
before deleting, but a put that has landed while its index record is not yet
installed is invisible to that check, and t4 applies puts and tombstones by
LSN, so the later `remove` wins and deletes live bytes.

Closing it needs a per-write generation in the store key, which also makes
`DiskResidue::superseded` unreachable and removes the in-place-overwrite case.
Not yet done. Do not re-report it as new.
