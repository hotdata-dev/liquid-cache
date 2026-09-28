# Working in this fork

`hotdata-dev/liquid-cache` is a fork of `datafusion-contrib/liquid-cache`.
`main` contains upstream's history in full plus our patches, so
`git merge-base main upstream/main` resolves.

## Set your remotes up first

The standard fork layout, which everything below assumes:

```
origin     hotdata-dev/liquid-cache          ours, where we push
upstream   datafusion-contrib/liquid-cache   theirs, read-only
```

See what you have first — the two cases need different commands, and running
the wrong one leaves both remotes pointing at this fork, where `upstream/main`
silently means our `main` and every recipe below is wrong with no error.

```
git remote -v
```

**Cloned this fork** (`origin` already correct) — add the other:

```
git remote add upstream https://github.com/datafusion-contrib/liquid-cache.git
```

**Cloned upstream** (`origin` points at *upstream*, which has already been
misread as the opposite of what it meant) — rename it, then add ours:

```
git remote rename origin upstream
git remote add origin git@github.com:hotdata-dev/liquid-cache.git
```

Either way, confirm the two point at *different* repositories before relying
on anything below:

```
git remote get-url origin    # must be hotdata-dev
git remote get-url upstream  # must be datafusion-contrib
```

Then `origin/main` and `upstream/main` are stable refs that mean one thing.
Prefer them over `FETCH_HEAD`, which holds only the most recent fetch and has
repeatedly produced instructions in this file that silently checked the wrong
thing.

`AGENTS.md` and `README.md` are upstream's and describe the project itself.
This file is ours and describes only what differs here. Nothing in this repo
should edit an upstream-owned file to record a fork convention: that conflicts
on every sync. Add a file upstream does not have instead.

## Branches

- `git fetch origin` before branching, and branch from `origin/main`. A local
  `main` goes stale with no signal, and branching off a stale one silently
  drops everything merged since.
- **Upstream PRs branch from upstream**, named `upstream/<topic>`. A branch cut
  from our `main` carries the whole patch stack into the PR diff.

  ```
  git fetch upstream
  git checkout -b upstream/<topic> upstream/main
  git log --oneline upstream/main..HEAD   # must list only your own commits
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

```
git fetch --multiple origin upstream
git checkout -b sync/upstream-<date> origin/main
git merge upstream/main
```

Raise a PR; do not use GitHub's "Sync fork" button, which offers to discard our
commits when the merge is not a fast-forward.

**Merge that PR, do not squash it.** A squash gives the result a single parent,
so upstream's history never enters `main`'s ancestry: the merge-base does not
move, the same upstream commits stay missing, and nothing reports it. Verify
afterwards — the merge happens on GitHub, so fetch before checking:

```
git fetch --multiple origin upstream
git merge-base --is-ancestor upstream/main origin/main && echo ok
```

A change we contributed upstream comes back as their squash of it. The content
matches but the commit does not, so the merge conflicts where both sides
touched the same lines — typically a module list each side appended to. Keep
ours *for the returned change*, and keep any other upstream edit in the same
hunk: upstream may have appended something of its own next to it, and taking
the whole hunk from our side drops that silently. Read the hunk rather than
resolving by rule.

Sync promptly rather than letting such a conflict wait: alone it is obvious,
bundled with real upstream work later it is not.

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
