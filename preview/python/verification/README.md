# Current dev5 installed-only verification

Candidate imports resolve to installed site-packages, with RECORD ownership and
hash checks. Frozen baseline/oracle modules remain local test-only references.
See ../../REPRODUCE.md for runnable commands and ../../VERIFICATION.md for scope.
../../VERIFICATION_SNAPSHOT.json binds the current final test/harness files.

Inherited tests include deliberate extreme-range contract updates, not an
unchanged-baseline claim. The original pre-integration copy manifests and
machine-local logs are omitted. Runtime code hashes are in SOURCE_SNAPSHOT.json
at the preview root and in each package's SOURCE_MANIFEST.json.
