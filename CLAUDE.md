# Working in this fork

`hotdata-dev/liquid-cache` is a fork of `datafusion-contrib/liquid-cache`.
`main` contains upstream's history in full plus our patches, so
`git merge-base main <any upstream commit>` resolves.

Nothing here states where upstream currently is, or where we are: both move.
Ask git instead.

```
git fetch https://github.com/datafusion-contrib/liquid-cache main
git log --oneline FETCH_HEAD..main   # ours that upstream does not have
git log --oneline main..FETCH_HEAD   # upstream's that we do not have
```

`AGENTS.md` and `README.md` are upstream's and describe the project itself.
This file is ours and describes only what differs here. Nothing in this repo
should edit an upstream-owned file to record a fork convention: that conflicts
on every sync. Add a file upstream does not have instead.

## Remotes

**Check before using a remote name. They vary by clone and they are not what
you would guess** — in at least one working copy `origin` is *upstream* and the
fork is a second remote named `fork`, which is the reverse of the usual
arrangement.

```
git remote -v
```

Commands below name repositories by URL rather than by remote, so they are
correct in any clone. Do the same when writing instructions for anyone else: a
bare `origin/main` is ambiguous here and has already been misread as the
opposite of what it meant.

## Branches

- Work off `main`, and fetch before branching — a local `main` goes stale with
  no signal, and branching off a stale one silently drops everything merged
  since.
- **Upstream PRs branch from upstream, not from `main`**, and are named
  `upstream/<topic>`. A branch cut from `main` carries our whole patch stack
  into the PR diff.

  ```
  git fetch https://github.com/datafusion-contrib/liquid-cache main
  git checkout -b upstream/<topic> FETCH_HEAD
  ```

  Before pushing, this must list only the commits you wrote — anything else is
  a fork patch that would land in the upstream diff. Re-fetch upstream on the
  line above it: `FETCH_HEAD` holds whatever the last fetch wrote, so after
  fetching any other remote it is no longer upstream and the check hides every
  fork patch.

  ```
  git fetch https://github.com/datafusion-contrib/liquid-cache main
  git log --oneline FETCH_HEAD..HEAD
  ```

  `.github/workflows/upstream-branch-guard.yml` checks the same property on
  push, before a PR exists. It tests where the branch diverged rather than
  whether it contains `main`'s current tip, because `main` moves with every
  merge and a branch cut from it last week contains today's tip nowhere.

- Before raising an upstream PR, reproduce the bug **on upstream's tree** —
  apply the test alone, watch it fail, then apply the fix. A fix whose code
  still exists upstream is not evidence the bug does; that mistake has cost us
  a withdrawn PR. Several of our patches repair machinery upstream does not
  have (`DiskResidue`, `reclaim_orphaned_disk`, `settle`) and are not
  upstreamable at all.

## Syncing from upstream

Merge upstream into a branch off `main` and raise a PR; do not use GitHub's
"Sync fork" button, which offers to discard our commits when the merge is not
a fast-forward.

```
git fetch https://github.com/datafusion-contrib/liquid-cache main
git checkout -b sync/upstream-<date> main
git merge FETCH_HEAD
```

A change we contributed upstream comes back as their squash of it. The content
matches but the commit does not, so the merge conflicts where both sides
touched the same lines — typically a module list that each side appended to.
Resolve by keeping ours, which already contains the change. Sync promptly
rather than letting such a conflict wait: alone it is obvious, bundled with
real upstream work later it is not.

## Building and testing

- **Pin the toolchain: `cargo +1.96.0 ...`.** There is no
  `rust-toolchain.toml`, so a bare `cargo` uses whatever default is installed,
  and an older one fails resolution with a wall of `requires rustc 1.9x` lines
  naming `vortex-*` and `sysinfo`. Those lines state the minimum each crate
  wants; use a toolchain at least that new.
- **Shuttle tests need a filter**: `cargo +1.96.0 test -p liquid-cache
  --features shuttle --lib shuttle_`. The feature swaps `crate::sync` to
  shuttle primitives for the whole test build, so running it unfiltered fails
  every test that touches a lock outside a shuttle runner, with "Are you
  accessing a Shuttle primitive outside of a Shuttle test?". That is by design,
  not a regression.
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
