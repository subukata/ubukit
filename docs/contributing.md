# Development and verification

Build from `python/` and `javascript/`; see [getting started](getting-started.md). Test installed wheel/sdist/tarball artifacts so repository imports cannot hide packaging mistakes. Keep Python runtime files under `python/src/ubukit/`; JavaScript runtime files are explicitly listed in package.json. Do not add research code or build outputs to either package.

Current correctness layers:

1. Language-focused unit and numerical tests in `python/tests/` and `javascript/tests/`
2. Installed ownership/hash guards and inherited regression suites in `python/verification/` and `javascript/validation/`
3. Cross-language fixture integrity in `tests/contracts/` and `tests/integration/`
4. Local artifact/manifest checks and manual cross-platform harness in `tools/ci/`

Run the source/verification snapshot refresh only after reviewing intended changes. It checks current runtime bytes against each package's existing source manifest; it must not bless a changed kernel automatically. The CI source policy is deliberately unselected until a reviewed immutable source commit and snapshot hashes are selected. A checkout of arbitrary HEAD is not authorization to run it.

The GitHub workflow remains manual, one OS per approved dispatch, with an explicit budget confirmation and short evidence retention. The account-wide $0 Stop usage budget is an external setting and must still be checked. This layout change does not run GitHub Actions or GPU work.

Historical test oracles that active suites import remain read-only references. Archive-only tests are older version-specific checks, independent research experiments, or duplicate legacy-root suites; their original bytes remain restorable. See the companion complete migration and test-inventory manifests before removing any active contract.

Project contributions use MIT. Preserve the [license scope](../LICENSE-SCOPE.txt) and all third-party notices. See [license review](license-review.md) for source-preservation records, retained notices and publication gates.

Migration notes, recorded debugging cases and earlier comparison results are
retained in [preview development history](development-history.md). They describe
their recorded snapshots and are not validation of a newly built artifact.
