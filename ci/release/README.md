# Private cross-platform candidate validation (manual staged execution)

## Exact candidate and scope

- Repository: `subukata/ubukit` (private)
- Source commit: `934b506cd6250e1e9b9ea5ce2a419330f29de487` (PR8 facade version correction included)
- Python `0.0.0.dev5`; JavaScript `0.1.0-dev.6`
- The initial CI introduction added `.github/workflows/private-candidate.yml` and `ci/release/**`. The follow-up portability correction also updates `preview/python/verification/runtime_tests/som_kmeans_extension/test_scipy_blas.py` and its entry in `preview/VERIFICATION_SNAPSHOT.json`. No runtime files, frozen oracles, publishing configuration, licenses, visibility, secrets, or billing settings change
- The owner approved private integration and staged execution on 2026-10-02. This workflow is manual-only; it never automatically starts the next OS stage

The workflow checks out its own harness/verification commit into `harness/` and the reviewed artifact source into `candidate/`. Python tests execute from the workflow-bound verification checkout, whose complete snapshots are verified and whose runtime SOURCE_SNAPSHOT must be byte-identical to the frozen candidate. This permits reviewed test-only portability corrections while reusing exactly the same wheel/sdist/tgz. Both checkout SHAs and the artifact source binding are verified. The artifact source verifier fails closed on any other candidate SHA. Git's process-local `core.autocrlf=false` preserves byte hashes on Windows

## Staged execution

Each manual dispatch selects exactly one OS with `stage`: `linux`, `windows`, or `macos`. No matrix or automatic next-stage trigger exists. No push, pull-request or schedule trigger; no automatic retries.

Start `linux` with blank `artifact_run_id`: one Ubuntu build (10-minute timeout) plus one Linux consumer (45-minute timeout). Review results and actual quota before dispatching `windows` with that Linux run ID, then review again before `macos`. Later stages download the same frozen artifact using read-only Actions access and do not rebuild. If the artifact expired, stop and reassess instead of silently rebuilding. A new Linux build is the only way a blank run ID can proceed. The three manually selected consumer targets are:

| Standard runner | Architecture | Python | JavaScript | Actual browser engines |
|---|---|---|---|---|
| ubuntu-24.04 | x64 | 3.12 base wheel/sdist plus Linux full wheel regression | Node 24, installed tgz suite | Playwright Chromium, Firefox, WebKit |
| windows-2022 | x64 | 3.12 base wheel/sdist focused/API/examples | Node 24, same installed tgz | Playwright Chromium, Firefox, WebKit |
| macos-15 | ARM64 | 3.12 base wheel/sdist focused/API/examples | Node 24, same installed tgz | Playwright Chromium, Firefox, WebKit |

Across all three separate dispatches, these configured job timeouts total 145 raw runner-minutes before any rounding/post-job overhead. The first Linux dispatch alone is capped at 55 configured job-minutes; Windows and macOS each add one 45-minute job only after a separate quota/result check. This is a resource envelope, not a promised duration, billing estimate, or remaining-allowance calculation. Current official documentation gives different OS prices; do not reuse historical minute multipliers without checking current terms

For planning only, the historical 1× Linux / 2× Windows / 10× macOS accounting would yield `(10+45)×1 + 45×2 + 45×10 = 595` included-minute equivalents. The currently published dollar rates yield `(55×0.006 + 45×0.010 + 45×0.062)/0.006 = 595` Linux-price-equivalent minutes for this equal-length Windows/macOS matrix. These coinciding estimates are not verification of the current included-minute debit algorithm: the current official billing page no longer states that multiplier table. Reserve additional room for per-job rounding and cleanup, and inspect actual included-usage changes after the run. Never treat 145 raw job-minutes as 145 deducted allowance minutes

A single build produces wheel, sdist and npm tarball with SHA-256 manifests. Every OS consumes those identical files. Frozen build artifacts are capped at 4 MiB; reports are capped at 10 MiB per OS, with one-day retention. The total proposed artifact envelope is under 35 MiB including small metadata. No Actions cache is enabled; venvs, browser binaries and node_modules are never uploaded

## Required cost preflight before each dispatch

Read-only observation on 2026-10-02 showed the account's current-month Actions panel at 0 of 2,000 minutes and 0 of 0.5 GB storage used. The existing account-wide Actions budget was $0 with Stop usage enabled. These were directly observed, not inferred from the plan. No setting was changed

Immediately before executing:

1. Recheck https://github.com/settings/billing and https://github.com/settings/billing/budgets for account `subukata`
2. Confirm sufficient included usage remains and the account-wide Actions product budget is still $0 with Stop usage enabled. If not, stop; do not change spending limits or use a paid runner
3. Confirm no other concurrent work could invalidate the usage snapshot. The allowance is account-wide. Artifact usage can be delayed by 6–12 hours
4. Select the pinned source SHA, the single next stage, and explicitly set `free_budget_rechecked=true`. For Windows/macOS supply the successful Linux artifact run ID. Do not advance until the preceding result and actual quota are reviewed
5. If GitHub refuses the run because a quota/budget is exhausted, report that block. Do not lift the limit or automatically retry

The boolean input only records a human preflight. It is not a billing API check and cannot enforce account billing settings. The already configured GitHub hard stop is the cost control; this workflow neither reads private billing credentials nor creates them

Official references: [Actions billing](https://docs.github.com/en/billing/concepts/product-billing/github-actions), [budgets](https://docs.github.com/en/billing/how-tos/set-up-budgets), [runner rates](https://docs.github.com/en/billing/reference/actions-runner-pricing), [standard runner labels](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)

## Gate details

- Portable source verification: all 89 runtime hashes, exact runtime path sets and 463 committed verification hashes; confirms the PR8 facade correction without changing source
- Python: independent clean wheel and sdist venvs, binary runtime dependencies, pip check, source ownership/hash checks, focused tests, 113 high-precision oracle cases, facade and both examples. Linux additionally runs inherited runtime/external harnesses and new tests. Build backend constraints prevent an unconstrained setuptools upgrade during sdist installation
- JavaScript: offline, scripts-disabled installation of the same frozen tgz; 45 test files; package documentation, VM/worker, SOM VM, differential and efficiency-worker stages
- Browser: each engine runs SOM/worker/routes/cache checks normally, with WebAssembly unavailable, and with CSP forbidding WASM compilation; normal mode requires real WASM assignment/finalization with no fallback. A separate page executes the sklearn fixture sweep and external-metrics module worker. Each scenario records failures and keeps testing the remaining engines; failures are never skips
- Tooling: official actions pinned to verified immutable commit SHAs. Playwright `1.63.0` and its three-package dependency tree are locked through normal `npm install --package-lock-only --ignore-scripts`; no fabricated lockfile. Playwright remains test-only

## Explicitly not established by this staged plan

No native Safari, installed Chrome/Edge channels, Intel macOS, additional architectures, Python 3.10/3.11/3.13/3.14, Node 20/22/26, or optional Numba claims. WebKit is not Safari. No demo UI/visual/accessibility acceptance or every-algorithm browser suite is promised. Python's lower declared dependency bounds remain separate compatibility work. Full dependency reproducibility across OSes is not claimed: runtime pins and key build tools are constrained, while transitive/platform dependencies are resolved and captured in per-install freeze logs

A CI pass is not publication or release approval. Licensing/provenance/public support claims remain their own release gates

## Overflow portability correction and independent evidence

The extreme overflow fixtures can reach either the public preflight rejection or the ordered-distance kernel rejection depending on OpenBLAS CPU dispatch. The public regression requires one of the two exact known safety messages and exact frozen-oracle/candidate parity for each fixture. A separate direct-kernel regression retains the exact ordered overflow message and explicitly checks an overflowing nonwinning center under default certification and forced unaudited fallback. Runtime code and frozen oracles are unchanged. Installed-package reports include processor/BLAS library information.

After source verification succeeds, JavaScript and browser gates are independent of a Python test failure. A failed Python gate still fails the overall workflow; it no longer hides unrelated JavaScript/browser evidence. Source verification failure or cancellation prevents those gates.
