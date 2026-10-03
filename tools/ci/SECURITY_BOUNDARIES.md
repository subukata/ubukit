# Security-boundary candidate: alpha.3

This private, unpublished preparation uses `ubukit==0.1.0a3` and
`ubukit-js@0.1.0-alpha.3`. It does not replace the source or exact archives of
alpha.2. A new source commit and a separate reviewed harness selection must be
bound before a fresh build and qualification. Null policy and exact-tree pins
intentionally prevent this preparation from dispatching against alpha.2.

## Changes and compatibility

- `.gitattributes` fixes text checkout bytes to LF, including preserved WAT,
  while preserving binary assets. Conversion-enabled Git checkouts must pass
  the same raw-byte verification; the verifier never normalizes inputs to hide
  differences. An offline Git regression includes the formerly failing WAT,
  all preserved source records and binary assets.
- Archive verification validates exact file inventory and bounded metadata,
  decompression and member reads before accepting payloads. It is a strict
  verifier for these small release archives, not a general-purpose extractor.
- Current documentation distinguishes the prior three-OS alpha.2 qualification
  from this new candidate, states registry publication status, and describes
  service-side input and process resource limits. `SECURITY.md` is a reporting
  and deployment policy draft, not evidence that private reporting is enabled.
- No numerical algorithm, API signature, default allocation behavior or declared
  interpreter/runtime dependency floor changes. The three Python version
  constants are the only changed runtime source bytes; JavaScript runtime and
  WASM bytes are unchanged. Large FCM cluster counts remain caller-controlled.
- The original historical JavaScript reference clones remain historical, outside
  runtime packages. They are not authorized replacements for the current safe
  clone helper. Historical development evidence is not mass-rewritten.

The source `.gitattributes` and `SECURITY.md` are now included in the verification
snapshot. Runtime manifests record the version-only Python changes explicitly.
Registry publication, repository visibility, branch protection, reporting
settings, credentials and asset ownership/license decisions remain separate.

## Strict archive profile

The verifier accepts only the exact reviewed package-file allowlists. Each input
archive is capped at 4 MiB compressed, with at most 512 entries, 4 MiB per file,
16 MiB total file payload and a 200:1 expansion ratio. ZIP central-directory
metadata is limited to 512 KiB and counted before the standard reader builds its
inventory. ZIP local headers, data descriptors, decompressed lengths and CRCs
must agree; only stored/DEFLATE regular files are accepted. Nonempty ZIP extra
fields (including Unicode path/ZIP64 extensions) are unsupported and rejected
after bounded framing checks, before payload reads.

Gzip TAR preflight reads bounded physical headers directly, instead of allowing
an archive reader to expand hidden GNU/PAX records without prior limits. Only
regular files, required parent directories and a small supported local-PAX subset
are accepted. It caps total expansion at 20 MiB (with a 64 KiB ratio allowance
for tiny padded TARs), physical headers at 1,024, individual PAX metadata at
4 KiB, total PAX metadata at 64 KiB and terminal padding at 10 KiB. Links, sparse
files, special files, unknown extensions, duplicate/unsafe names and excessive
padding are rejected. Reading later TAR headers necessarily decompresses skipped
file spans within these bounds; package payloads are materialized only after the
full exact inventory and trailer pass. No archive member is extracted to disk.

These intentionally strict limits fit the current source archives with ample
headroom. A future larger package or different archive format requires an
explicit reviewed verifier change rather than silently relaxing the limits.
They bound this verification operation, not package runtime process resources.

## Most recent exact-artifact evidence

The prior alpha.2 source was `464969206fc336723b6f33f0ccaf70cf2110136d`;
its verification harness was `3a638f8b4f2df24bd382a4f879a438c3a8ad3d73`.
The Linux build and three defined OS stages succeeded:
[Linux](https://github.com/subukata/ubukit/actions/runs/37085466524),
[Windows](https://github.com/subukata/ubukit/actions/runs/37086270594),
[macOS](https://github.com/subukata/ubukit/actions/runs/37086865574).
These links may require repository access. The extended Python regression was
Linux-only; optional Numba and native Safari were not covered. Windows testing
used the same Linux-built archives and `core.autocrlf=false`. It did not validate
a default CRLF-converting source checkout.

No results from those runs qualify the changed alpha.3 archive bytes. Record
new local and remote results with exact source/harness identities and artifact
hashes. Only an independently verified source/harness selection can activate
`exact-alpha.json`; do not reuse old selection pins or archive hashes.
