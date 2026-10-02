# Exact alpha source and artifact handoff

This configuration is private, manual-only and unselected. It performs no
publication. It qualifies the proposed `ubukit==0.1.0a1` and
`ubukit-js@0.1.0-alpha.1` only after their identities and source are reviewed.
Name availability/ownership, public destinations, support claims and publication
approval are separate decisions.

## Frozen source, separate verification harness

`exact-alpha.json` binds the entire 618-file frozen alpha Git tree, including file
modes, to `6d6d7d875058be3ee9aa8db8b001df7657e0f52f`. Its package-input snapshot
SHA-256 is `eb6807035e64d369b8317f34d97c667f548917685f22addd3bdee29d005c9049`.
These are content identities, **not an existing commit or CI run**. No source
commit is invented. The policy's four source/build pins remain null in this
preparation.

The existing preview source `478fbc25266ea60a5dcb17c58c8cca364fa66225` and its
artifact run `37056004329` are different identities. Their CI results and archives
cannot certify the alpha. Do not copy preview policy pins into this harness.

The source commit and harness commit deliberately differ. A source commit keeps
the frozen 618-file alpha tree unchanged, including its null policy. A later
harness commit contains this exact-alpha configuration and a reviewed selection.
Runtime, package metadata, licenses/notices and `SOURCE_SNAPSHOT.json` remain
identical between the two checkouts. The harness snapshot changes, as it must.

If the final source needs any further change, stop this exact-tree selection.
Review and freeze a new candidate rather than relabeling the old one or altering
its records to make verification pass.

## Prepare the selection after a real source commit exists

1. Under the appropriate repository-write authorization, commit the frozen source
   and verify the remote commit's complete tree is the tree above. A local test
   commit, a branch name, a short SHA or an assumed future SHA is insufficient.
   Preserve already tracked evidence even when current ignore patterns match it;
   a new `git add .` can otherwise omit the two historical WASM result files.
   The exact-tree guard rejects such an incomplete source.
2. Materialize that immutable source commit in a clean `candidate/` checkout.
   Materialize the separately reviewed configuration in `harness/`. Check that
   the latter still has all four policy pins null. No runtime/package change is
   permitted in the harness.
3. From outside both checkouts, run the following with the real 40-character
   source SHA. This is an offline preparation command, not a dispatch:

   ```sh
   python harness/tools/ci/scripts/exact_alpha.py select \
     --source candidate --harness harness \
     --source-sha "$REVIEWED_ALPHA_SOURCE_SHA" --output alpha-selection
   ```

   The helper checks Git cleanliness/identity, full tree/modes, both complete
   snapshots and both package identities. It writes a new external directory
   containing only two repository overlay files plus `selection-receipt.json`.
   Inputs are never changed. Existing output directories are never overwritten.
   The receipt explicitly states that it has not verified remote existence.
4. Review the emitted policy and verification snapshot. Copy only
   `tools/ci/candidate-policy.json` and
   `tools/provenance/VERIFICATION_SNAPSHOT.json` into the harness; keep the receipt
   outside the checkout. Both artifact-reuse pins must remain null for the first
   build. Do not regenerate package baselines or the product snapshot.
5. Run the offline unit/static checks below, then commit the complete reviewed
   harness. Verify the resulting remote tree and immutable harness SHA. Selection
   is not validation success, a spending authorization or publication approval.

## Build once and consume exactly that bundle

Only after the required fresh account-wide $0/Stop-usage and remaining-allowance
checks, and applicable action authorization, may the existing manual workflow
run. The receipt starts `free_budget_rechecked` at false; a selection script must
never attest to billing state.

- First run: `stage=linux`, `candidate_sha=<reviewed exact alpha commit>`, blank
  `artifact_run_id`. The workflow's exact-alpha guard checks the frozen tree
  before building and before any installed/browser consumer.
- The unchanged build script uses pinned Python build tools to create a wheel,
  sdist and npm tarball once. `build-manifest.json` records the exact source and
  build-harness commits, source/harness snapshots, repository, run ID and artifact
  hashes. It also binds each archive inventory to the reviewed source bytes.
- Every consumer rejects wrong source, run, repository, build-harness provenance,
  missing/extra archive members, changed contents and unexpected bundle files.
  It consumes the frozen bundle, never a registry copy or replacement release
  archive rebuilt from the checkout. The sdist consumer deliberately performs
  the normal isolated PEP 517 installation build from that exact frozen sdist.
- Review Linux results before deciding on another OS. Each later selected stage
  consumes the successful Linux run ID after a fresh cost preflight. If the
  harness commit changed, stop unless both reuse pins explicitly name the
  reviewed build-harness SHA and verification-snapshot digest.
- Save results and bundle before the configured one-day artifact expiration.
  Expiration does not permit silent rebuilding. A rebuild produces a newly
  identified bundle and must be qualified separately.

Build commands, pinned tools and source selection are repeatable. Byte-identical
archives across separate builds, timestamps, tool patch versions or environments
are **not claimed**. The release binding is the saved, exact CI-tested bundle.
Fresh CI-built archive hashes may differ from the earlier local provisional
archives. Matching sources do not prove those earlier archive bytes passed CI.

## Offline checks

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tools/tests -v
python tools/ci/scripts/verify_source.py .
actionlint .github/workflows/private-candidate.yml
```

The configuration must initially fail closed for an unselected policy. Fixture
tests use clearly local temporary Git commits; they are not remote integration
or CI evidence. A prepared two-file selection must preserve the product snapshot
and satisfy the full harness snapshot after application.

## Coverage and publication gate

The unchanged workflow selects Python 3.12 / Node 24 on one standard runner per
dispatch. It can run clean wheel/sdist consumers, the complete configured Node
suite, and Playwright Chromium/Firefox/WebKit normal and fallback modes. A browser
launch is not considered executed until scenarios actually run. WebKit is not
native Safari. The configuration alone qualifies no platform or browser.

Minimum declared Python 3.10 / Node 20, lower dependency bounds, optional Numba,
other OS/architecture combinations and broader browser/algorithm claims remain
outside the initial local evidence. Either qualify a chosen support profile or
limit claims to the exact passing evidence. Do not silently widen this workflow.

Before publication, review the final successful source/harness/run identities,
all three exact artifact hashes, all intended support results and remaining
limitations. Confirm registry ownership, names, versions, npm distribution tag
and destination/visibility separately. Neither this helper nor this workflow
uploads to a registry, changes repository visibility or grants publication
permission.
