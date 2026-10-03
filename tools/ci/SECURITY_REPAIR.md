# Security repair candidate: alpha.2

This new, private candidate is `ubukit==0.1.0a2` and
`ubukit-js@0.1.0-alpha.2`. It supersedes the previously tested alpha.1 source for
new qualification; it does not relabel, replace or certify the alpha.1 archives.
No registry publication or repository visibility change is part of this work.

## Reviewed changes

- The repository demo server exposes only named browser pages/scripts and the
  28 reviewed runtime modules. It binds to loopback, validates local Host and
  same-origin Origin headers, accepts GET/HEAD, rejects symlinks and unlisted
  paths, and sends `nosniff`. It is not a sandbox for hostile local writers.
- The JavaScript owned-object clone defines own data properties without invoking
  inherited `__proto__` setters, while retaining supported object/array cycles.
- Missing optional Numba errors name the actual `ubukit[numba]` distribution
  extra. Numerical implementations and the optional extra itself are unchanged.
- Verification uses pip 26.2, pytest 9.0.3 and setuptools 83.0.0 from reviewed
  official PyPI version metadata. The build backend floor is setuptools 83.0.0.
  Python >=3.10 / Node >=20 declarations and runtime dependency floors are unchanged.

## Toolchain and dependency binding

`constraints/installer.txt`, `build-py312-locked.txt`, `test-py312-locked.txt` and
`base-py312-locked.txt` pin complete base/test/build verification dependencies
with SHA-256 hashes and binary-only installation. Numerical wheel hashes cover
CPython 3.12 wheels for platform selection; recording a wheel is not evidence
that its platform passed. `toolchain-provenance.json` records official metadata
URLs, filenames, sizes and hashes. Hashes prove byte identity to that reviewed
metadata, not absence of unknown vulnerabilities.

Each fresh wheel/sdist consumer first upgrades pip from the hash-bound installer
lock. It installs locked runtime/test wheels, downloads a hash-verified build
wheelhouse, then installs the frozen local archive with no dependencies. The
sdist still exercises PEP 517 isolation, with network indexes disabled and only
the verified wheelhouse available. `build-tools.txt` is a version-only build
constraint; it is not the active installation requirements file.

Historical files under `python/constraints/requirements-*-verified.txt` remain
unchanged records of earlier stacks. They are not active alpha.2 locks. Optional
Numba uses the separate legacy opt-in path and is outside this base profile.
No claim is made that those historical or optional stacks have been updated.

## Immutable qualification

Use the source/harness procedure in `EXACT_ALPHA.md`, with a newly reviewed
alpha.2 full tree, file count, package snapshot and version identity in the
harness's `exact-alpha.json`. The source checkout retains an unselected policy.
A later harness commit selects the real source commit and requires a fresh Linux
build. Both artifact-reuse pins are null when building for the first time.
Reusing that exact successful bundle on Windows/macOS requires unchanged harness
identity (or separately reviewed build-harness pins), a fresh free-quota check
and the successful Linux run ID. Preserve each run's evidence and exact bundle.

Local focused/security tests are not the final qualification. Fresh installed
wheel/sdist, complete configured JavaScript suite, three-engine installed-package
cases and the actual hardened demo-server smoke run on selected standard runners.
Record actual results per OS/engine; do not reuse old alpha.1 pass claims.
Native Safari, minimum interpreter versions, optional Numba and other architectures
remain outside this profile unless separately run and documented.
