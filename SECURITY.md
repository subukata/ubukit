# Security policy (draft)

At source preparation on 2026-10-03, UbuKit was unpublished alpha software.
This dated draft records the security
boundaries and a proposed reporting procedure; the maintainer still needs to
confirm a private reporting contact. No response-time or maintenance-lifetime
commitment is made here, and this file does not enable any repository security
setting.

## Versions and tested profile

The locally prepared candidates are `ubukit==0.1.0a3` and
`ubukit-js@0.1.0-alpha.3`. Their exact-artifact qualification and initial PyPI/npm
publication were pending at preparation. Check the maintainer’s release record
for subsequent results and exact archive hashes. No stable or long-term-supported
release line existed at preparation. Report issues against the current candidate
or identify the exact earlier commit/archive affected; older previews are not
promised backports.

The most recent completed qualification is alpha.2, source
`464969206fc336723b6f33f0ccaf70cf2110136d` and harness
`3a638f8b4f2df24bd382a4f879a438c3a8ad3d73`. It consumed the same frozen wheel,
sdist and npm tarball on Linux x64, Windows x64 and macOS ARM64 using CPython
3.12 / Node 24. Full Python wheel regression ran only on Linux; no optional Numba
backend ran. JavaScript included bounded actual Playwright Chromium, Firefox and
WebKit execution. WebKit is not native Safari. See the exact run links, versions
and exclusions in the [coverage boundary](docs/numerics.md#coverage-boundary).
Those results do not certify changed alpha.3 artifacts, arbitrary source
checkouts, all declared interpreter versions or production security.

## Reporting a suspected vulnerability

- Use a private maintainer contact you have independently verified. If GitHub
  shows a private **Report a vulnerability** control for this repository, that
  may be used after confirming the channel is available. This draft does not
  assert that GitHub private vulnerability reporting is enabled. A dedicated
  private reporting address has not yet been designated in this policy.
- If no verified private route is available, open only a non-sensitive issue
  requesting a private reporting channel. Withhold exploit details, personal
  data, credentials and other sensitive material from public issues or comments.
- Include the package version, exact source commit or artifact SHA-256, operating
  system, interpreter/browser and dependency versions, affected public API,
  expected/observed behavior, prerequisites and a minimal synthetic reproducer.
  Distinguish installed-artifact behavior from a source checkout or historical
  validation fixture. Explain impact and any known workaround without providing
  real user data or secrets.
- Keep testing bounded and local to systems you are authorized to test. For
  resource-exhaustion concerns, small inputs or an intercepted allocation can
  establish the relevant boundary without exhausting a machine or service.

## Deployment trust and resource boundaries

UbuKit is a numerical library, not a request sandbox. Valid finite numerical
inputs can still require excessive memory or CPU. Before conversion/allocation,
callers handling untrusted requests must cap payload bytes, dimensions and their
products, iterations/epochs, requested arrays/history, serialized output size,
queued work and concurrency. Python FCM has no universal cluster-allocation
ceiling; do not assume `K <= N` is enforced or a required mathematical rule.

Primary scratch limits do not bound all input conversions, distance/weight
arrays, membership matrices, snapshots, return copies, numerical-library
workspace or exceptional integer/BigInt temporaries. Run admitted work in an
isolated worker process with externally enforced memory, CPU and wall-time
limits; bound thread counts and terminate over-budget work. Browser Workers,
cooperative cancellation, `timeBudgetMs` and AMI work-term limits alone do not
supply that service boundary. See [resource guidance and an illustrative caller
policy](docs/numerics.md#resource-boundaries-for-untrusted-requests).

Optional Numba requires separate environment qualification. Keep its
`NUMBA_CACHE_DIR` private and trusted, writable only by the executing trusted
account, never in an upload directory or a shared attacker-writable location.
Do not load untrusted pickle files or compiler caches. This is a configuration
trust requirement; no new cache exploit is established by this policy.

The repository demo server is for loopback development. Its path allowlist and
Host/Origin checks are not authorization for public deployment or a sandbox
against hostile local filesystem writers. Historical validation fixtures are
not supported application entrypoints. Use the public package APIs and reviewed
local artifacts, and retain their exact hashes when reporting findings.
