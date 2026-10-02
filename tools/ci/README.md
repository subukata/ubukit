# Exact alpha candidate validation

This manual-only harness builds wheel, sdist and npm tarball once, then consumes the same frozen artifacts in separately approved OS stages. It does not publish packages, enable paid runners, change billing settings, or start the next stage automatically.

See [EXACT_ALPHA.md](EXACT_ALPHA.md) for the frozen-alpha identity, two-commit selection process and exact-artifact release handoff. The workflow remains private and manual-only.

## Current state: intentionally blocked

`candidate-policy.json` has no approved alpha source SHA or source-snapshot digest. The exact alpha is therefore not dispatch-ready. A dispatch input cannot approve a candidate, select `HEAD`, or substitute any other arbitrary revision.

The preserved pre-migration source was `86321a9f1bc405ddb3251facfe1d876064c9c9dd`; the archived main snapshot was `5a197111b605c495f3f2a7ba69a6e09bb44e8caf`. Their test results remain historical evidence. They do not validate artifacts built from the new product layout, and the historical source cannot satisfy its new path contract.

Before any future execution, review and commit the product layout, select its exact immutable 40-character source SHA plus the SHA-256 of `tools/provenance/SOURCE_SNAPSHOT.json`, and update the policy in a separate reviewed harness commit. Execution still requires separate approval and the cost preflight below. Updating snapshots never changes this source approval policy.

## Source, harness and artifact binding

- `tools/provenance/PRODUCT_CONTRACT.json` declares package roots, runtime discovery rules, packaging metadata, Python facade exports, JavaScript entrypoints/algorithms, verification inputs and exclusions
- Per-package `SOURCE_MANIFEST.json` files preserve the approved runtime file hashes. The refresh tool refuses runtime drift rather than silently recomputing those baselines
- `SOURCE_SNAPSHOT.json` binds exact runtime paths and hashes, package versions, package allowlists and packaging/documentation bytes. Versions and file counts are derived from the explicit manifests; release scripts do not assume a particular development suffix or count
- `VERIFICATION_SNAPSHOT.json` binds the complete current test/harness input set. Build directories, environments, caches, generated WASM outputs, artifacts and result folders are excluded. Additional unlisted verification inputs fail validation
- The workflow checks out its own immutable `github.sha` into `harness/` and the separately approved source SHA into `candidate/`. Both Git revisions and clean worktrees are checked. Their complete source snapshots must be identical, allowing only reviewed verification-harness changes
- Both Python and JavaScript installed tests execute from the workflow-bound harness against artifacts built from the separately bound source. Python facade parity is retained
- `build-manifest.json` records the source SHA, independent build-harness SHA, source/harness snapshot digests, build run ID and exact artifact checksums. Consumers reject changed, missing or extra artifact files. Reusing artifacts built by a different harness requires explicit reviewed build-harness SHA and snapshot-digest pins in the policy; an input does not relax the lock

The artifact source includes `python/` and `javascript/`, with Python examples in `examples/python/`. There is no runtime code execution during source verification.

## Local preparation checks

From the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tools/tests -v
python tools/provenance/refresh.py
python tools/ci/scripts/verify_source.py .
python tools/ci/scripts/verify_artifact_contents.py --source . --artifacts /path/to/frozen-artifacts
```

Refresh only after reviewing the intended source and harness changes. It writes local snapshot metadata and does not build, install, dispatch, approve or publish anything. Fixture tests exercise correct and wrong source SHA cases, separate harness identity, dirty checkouts, manifest/path tampering and artifact provenance using temporary local files and Git repositories.

## Manually selected OS stages

The only trigger is `workflow_dispatch`; there is no push, pull-request, schedule, matrix, automatic retry or next-stage trigger.

1. After explicit approval and cost preflight, dispatch `linux` with a blank `artifact_run_id`. Ubuntu builds once (10-minute timeout) and performs the Linux consumer gate (45-minute timeout)
2. Review its results and actual allowance. Dispatch `windows` only after separate approval/preflight, supplying the successful Linux artifact run ID; this has one 45-minute consumer job
3. Review again before separately dispatching `macos` with the same frozen artifact run ID; this also has one 45-minute consumer job

If artifacts expire or a quota/budget blocks execution, stop and reassess. Do not silently rebuild, raise limits or retry. A blank artifact run ID can start a build only on Linux. Each dispatch chooses one standard runner: `ubuntu-24.04` x64, `windows-2022` x64 or `macos-15` ARM64; Python 3.12 and Node 24 remain configured.

The configured timeout envelope is 55 raw job-minutes for the first Linux stage and 45 for each later OS, totaling 145 before per-job rounding/cleanup. This is not a billing estimate or remaining-allowance claim. Check current account terms and actual usage rather than treating raw minutes as deducted allowance.

Frozen artifacts are capped at 4 MiB; reports are capped at 10 MiB per OS. Both have one-day retention. No Actions cache is enabled, and environments, browser installations and node_modules are never uploaded.

## Required $0 preflight before every dispatch

1. Recheck [account billing](https://github.com/settings/billing) and [budgets](https://github.com/settings/billing/budgets) for account `subukata`
2. Confirm sufficient included usage remains and the account-wide Actions product budget is still **$0 with Stop usage enabled**. If either is unconfirmed, stop; do not change limits or use a paid runner
3. Confirm concurrent work has not invalidated the allowance observation. Usage is account-wide, and storage accounting may be delayed
4. Review the exact pinned source, one selected OS, prior-stage results and artifact run ID. Only then set `free_budget_rechecked=true`
5. If GitHub refuses execution, report the block. Never lift a spending limit or automatically retry

The boolean input is a human preflight record, not a billing API or enforcement mechanism. The account hard stop is the cost control. No old quota observation is treated as current authorization.

## Execution coverage and limits

Python gates use independent clean wheel/sdist installations, constrained build tools and binary-only runtime dependencies, ownership/hash checks, the sole `ubukit` namespace, absent legacy aliases, exact facade exports, focused tests, high-precision oracles, API and examples. Linux additionally runs inherited runtime/external harnesses and new tests. Full dependency reproducibility across OSes is not claimed; installed versions are captured.

JavaScript uses offline scripts-disabled installation of the frozen tarball, discovered installed-package tests, package/API documentation contracts, VM and worker compatibility, SOM VM cases and differential/efficiency workers. The JavaScript gate remains independent of Python failure after source verification succeeds.

The browser gate runs Playwright Chromium, Firefox and WebKit in normal, WebAssembly-unavailable and CSP-blocked modes, plus the external-metrics fixture/worker page. Normal mode requires actual WASM execution. Failures remain failures rather than skips. WebKit is not native Safari.

This restructure and its local fixtures do not establish new Windows/macOS, GPU, native Safari, Chrome/Edge channel, Intel macOS, additional Python/Node version, optional Numba, UI/accessibility, lower-dependency-bound or every-algorithm browser coverage. A CI pass is not publication, license clearance or release approval.

Official billing references: [Actions billing](https://docs.github.com/en/billing/concepts/product-billing/github-actions), [budgets](https://docs.github.com/en/billing/how-tos/set-up-budgets), [runner pricing](https://docs.github.com/en/billing/reference/actions-runner-pricing)
