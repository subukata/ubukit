> Namespace dev6: this harness now imports implementation modules only under `ubukit._impl`; ownership guards require the installed `ubukit` tree. Frozen numerical fixtures/tolerances and oracle bytes remain unchanged. Current runtime hashes are in `staging/SOURCE_MANIFEST.json` relative to the Python preview root; historical hashes/results below are not new migration measurements.

# Current dev6 installed-only verification

Candidate imports resolve to installed site-packages, with RECORD ownership and
hash checks. Frozen baseline/oracle modules remain local test-only references.
See ../../REPRODUCE.md for runnable commands and ../../VERIFICATION.md for scope.
../../VERIFICATION_SNAPSHOT.json binds the current final test/harness files.

Inherited tests include deliberate extreme-range contract updates, not an
unchanged-baseline claim. The original pre-integration copy manifests and
machine-local logs are omitted. Runtime code hashes are in SOURCE_SNAPSHOT.json
at the preview root and in each package's SOURCE_MANIFEST.json.
