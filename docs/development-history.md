# Preview development history
This developer archive preserves migration, debugging and validation notes moved
out of the user guides. Each excerpt identifies its original immutable source.
Statements about “current”, “new”, checks, versions or paths refer to that source,
not this documentation candidate. Command blocks are historical instructions.
For current installation, use [getting started](getting-started.md).

This documentation-only candidate changes packaged README/guide bytes. Earlier
artifact or CI results do not qualify the new package metadata: fresh artifact
validation is still required. Runtime source, license notices, package versions,
CI policy and workflows are unchanged. No build, CI or GPU run is recorded here.

## Python namespace migration and source provenance

Source: [python/README.md:12–25](https://github.com/subukata/ubukit/blob/35389f9d9a101c5a8383d50e6a659fbed5b7543e/python/README.md#L12-L25)

This private `0.0.0.dev6` candidate consolidates every implementation under
`ubukit._impl`. The only installed top-level runtime package is `ubukit`.
The existing 38 facade exports, signatures, defaults, result contracts, stopping
rules and numerical implementations are retained. No new algorithm aliases or
backend defaults are introduced. The opt-in localized SOM-OLP implementation
from dev5 remains private and opt-in.

`SOURCE_MANIFEST.json` records every current runtime file and its mapped dev5
source/hash. Four inherited files have import/facade/documentation changes;
all other inherited runtime files retain their bytes. The only new runtime file
is the private package marker `ubukit/_impl/__init__.py`. Namespace migration
is not a new performance claim or a statement of cross-platform qualification.

## Python verification-bundle entry point

Source: [python/README.md:44–48](https://github.com/subukata/ubukit/blob/35389f9d9a101c5a8383d50e6a659fbed5b7543e/python/README.md#L44-L48)

If you received a private verification bundle instead, use the
[bundle installation example](https://github.com/subukata/ubukit/blob/35389f9d9a101c5a8383d50e6a659fbed5b7543e/python/README.md#inherited-numerical-qualification-and-installation-example)
below. Its paths are relative to the extracted bundle, not this README's directory.

## Python facade implementation inventory

Source: [python/README.md:55–57](https://github.com/subukata/ubukit/blob/35389f9d9a101c5a8383d50e6a659fbed5b7543e/python/README.md#L55-L57)

The `ubukit` root import is lazy and imports neither NumPy nor Numba. Its 38
exports still comprise 35 aliases and three thin sample-major adapters:
`fit_rcm`, `fit_exrcm`, and `assign_rcm`.

## Python private implementation, release gates and prior qualification

Source: [python/README.md:106–130](https://github.com/subukata/ubukit/blob/35389f9d9a101c5a8383d50e6a659fbed5b7543e/python/README.md#L106-L130)

The private RCM implementation retains `(K,N)` arrays and its original result
type. Only the public facade adapts memberships; private paths are not a public
legacy-compatibility promise. Retained dev5 artifacts are unchanged separately.

The established public entrypoints and sample-major membership axes remain.
See the companion correction report for the specific numerical and input-validation
changes. The runtime archive does not include JavaScript, datasets, benchmarks,
tests or logs.

## Scope and release gates

Local packaging tests are separate from numerical release approval. This private
candidate has no public release authorization. Multi-OS/Python and browser
checks, minimum dependency versions, final name/version policy, project license,
final public API/dependency policy, and explicit publication approval remain open.
The prior main-JavaScript FCM blocker is not a claim about this package or the
final selected JavaScript candidate. This is a Python-only artifact.

Declared minimum versions are resolver metadata, not a tested version matrix.
See the companion integration report for the exact tested environment and
passed, skipped, excluded, and unrun checks. Existing acceleration measurements
did not include the facade transpose-copy cost; no end-to-end API speedup is
claimed here.

## Python external-metrics namespace migration

Source: [python/README.md:145–150](https://github.com/subukata/ubukit/blob/35389f9d9a101c5a8383d50e6a659fbed5b7543e/python/README.md#L145-L150)

The formerly generic external-metrics modules now live under `ubukit._impl`.
Their optional Numba import is relative. No unrelated top-level module can
shadow those internal imports. The public facade still aliases the same scoring
functions without numerical wrappers.

## Python provenance, inherited environments and bundle installation

Source: [python/README.md:153–186](https://github.com/subukata/ubukit/blob/35389f9d9a101c5a8383d50e6a659fbed5b7543e/python/README.md#L153-L186)

Its `source_audit.json` is included in the verification
companion, under `verification/external_provenance/`. No project license is
chosen. Historical performance results are not new measurements of this wheel.

## Inherited numerical qualification and installation example

The inherited pre-migration private trial was checked on Linux x86_64 with CPython 3.12.14 only. Namespace-candidate checks are recorded in the separate migration report.
All integration environments used NumPy 2.3.5, SciPy 1.17.0 and threadpoolctl
3.6.0. Both scikit-learn 1.7.2 with optional Numba 0.63.1, and scikit-learn
1.8.0 with optional Numba 0.67.0 were checked. No Windows/macOS, ARM, browser,
or minimum-supported-version claim follows from these tests. Matching historical
version numbers does not carry the old benchmark results over to this facade.

From the extracted private verification bundle directory (containing `tools/`,
`artifacts/`, `examples/` and `constraints-final-stack.txt`), run the following
with Python 3.12 in a fresh environment. The preflight is read-only and is not a pip install hook. Stop on
a blocked result; do not co-install or automatically uninstall another owner.

```sh
python3.12 -m venv .venv-ubukit-preview
.venv-ubukit-preview/bin/python -I -B tools/check_install_environment.py
.venv-ubukit-preview/bin/python -m pip install -c constraints-final-stack.txt artifacts/ubukit_bundled_local_preview-0.0.0.dev6-py3-none-any.whl
.venv-ubukit-preview/bin/python -I -B examples/all_methods.py
# Optional Numba in this same isolated environment:
.venv-ubukit-preview/bin/python -m pip install -c constraints-final-stack.txt 'artifacts/ubukit_bundled_local_preview-0.0.0.dev6-py3-none-any.whl[numba]'
.venv-ubukit-preview/bin/python -I -B examples/all_methods.py --with-numba
```

Install the local wheel with the supplied companion constraints for this private
trial. They pin the tested stack rather than claiming all resolver outcomes
were validated. The source archive includes this README; the example script,
preflight, constraints and full verification harness are in the companion bundle.

## Python extreme-input regression history

Source: [python/README.md:285–294](https://github.com/subukata/ubukit/blob/35389f9d9a101c5a8383d50e6a659fbed5b7543e/python/README.md#L285-L294)

## Extreme-input acceptance

The earlier dev2 centroid correction exposed a squared-distance-underflow
rejection for an m=1000, D=128 fixture. The guarded finite-m implementation
below now supports the original and row-zero-shift versions of that fixture on
all five FCM backend selections. This supersedes the rejection-only policy; it
is not a promise of universal float64 accuracy or an unchanged extreme-input
trajectory. See the companion numerical report for verification and limits.

## Python trajectory evidence and optimization integration history

Source: [python/README.md:353–368](https://github.com/subukata/ubukit/blob/35389f9d9a101c5a8383d50e6a659fbed5b7543e/python/README.md#L353-L368)

Extreme-path trajectories are intentionally not bit-compatible with the prior
rounded-membership implementation. Near-coincident centers can turn tiny center
changes into large membership changes. The stabilization is not a guarantee
that every final membership is closer to a fully arbitrary-precision trajectory;
a 400-digit reference panel found improved centers but not uniformly improved
membership accuracy. Returned-pair consistency in that reference fixture and full-trajectory
accuracy are distinct checks. Membership/center consistency is not universal:
restoring a large origin can round the published centers while U retains the
working-coordinate update; the diagnostic flags this situation. See the repository preview numerical report and reference evidence.


## Combined private preview

The earlier dev3 integration added the optional standard-library-only optimization API, which remains unchanged. See [OPTIMIZATION.md](https://github.com/subukata/ubukit/blob/35389f9d9a101c5a8383d50e6a659fbed5b7543e/python/OPTIMIZATION.md). The existing scientific dependencies are unchanged; importing the optimization facade does not require them. The bundled SOM exceptional-range contract is documented in [NUMERICAL_LIMITS.md](https://github.com/subukata/ubukit/blob/35389f9d9a101c5a8383d50e6a659fbed5b7543e/python/NUMERICAL_LIMITS.md).

## JavaScript integration inventory and provenance

Source: [javascript/README.md:1–11](https://github.com/subukata/ubukit/blob/35389f9d9a101c5a8383d50e6a659fbed5b7543e/javascript/README.md#L1-L11)

# UbuKit JavaScript: dev6 private efficiency candidate

This repository snapshot contains the reviewed JavaScript candidate
`0.1.0-dev.6`, with 27 runtime files, seven package entrypoints and nine
algorithm/metric registry names. It includes
FCM/SOM extreme-range repairs, snapshot ownership and session validation fixes,
optional lightweight TPE/random optimization, ARI/AMI external metrics, and
traditional online SOM plus true BatchSOM. SOURCE_MANIFEST.json binds
the current runtime files to SHA-256 hashes.

## JavaScript packaged provenance and verification scope

Source: [javascript/README.md:181–189](https://github.com/subukata/ubukit/blob/35389f9d9a101c5a8383d50e6a659fbed5b7543e/javascript/README.md#L181-L189)

SOURCE_MANIFEST.json identifies the immutable input and every packaged runtime
hash. The separate integration verification bundle contains tests, comparison
snapshots, logs and a report; these are intentionally excluded from the npm tarball.
Testing does not imply new performance measurements, Windows/macOS/ARM coverage,
or real-browser verification. Consult the integration report for checks actually
run. Registry publication and deployment still require separate authorization.

## JavaScript browser-check environment blocker

Source: [javascript/README.md:227–229](https://github.com/subukata/ubukit/blob/35389f9d9a101c5a8383d50e6a659fbed5b7543e/javascript/README.md#L227-L229)

Node execution is tested from an installed tarball. The module has no Node-only
imports and is designed for modern browsers; an actual browser smoke run could
not be completed because the cloud browser blocked the local test URL.

## FCM near-coincident-center reference fixture

Source: [docs/numerics.md:27–36](https://github.com/subukata/ubukit/blob/35389f9d9a101c5a8383d50e6a659fbed5b7543e/docs/numerics.md#L27-L36)

Near-coincident centers are ill-conditioned: the existing m=64, seed994 fixture
changes memberships by as much as approximately 0.6505 versus dev2 even though
center changes are small. An independent 400-digit six-step trajectory improves
center accuracy in that case, but membership accuracy is not uniformly improved.
The detailed report and full reference evidence are retained. No arbitrary
precision, globally closer trajectory, global optimality, or cross-platform bit
identity is promised. Reinitializing from rounded public U cannot restore hidden
weights from an earlier run.

## SOM-OLP exact weighted-mean repair fixtures

Source: [docs/numerics.md:52–61](https://github.com/subukata/ubukit/blob/35389f9d9a101c5a8383d50e6a659fbed5b7543e/docs/numerics.md#L52-L61)

A narrow prototype-mean repair can also activate after ordinary inputs generate
tiny unit masses. One inherited JavaScript case generates mass around 1e-117:
its repaired W[5] is 0.6884290744771887, exactly the rounded independent 160-digit
weighted-mean reference, rather than the old 0.6884290744771885. A second inherited case generates a tiny unit mass that rounds to zero after
the update; its W[24] changes from 2.9543435536324973 to 2.954343553632498.
These two coordinates are tested against separately audited exact weighted-mean
references. All other entries, events, P/V/history, labels and iteration results
remain exact; the other 78 cases retain their original exact comparison.

## Exploratory Optuna comparison scope

Source: [docs/numerics.md:83–91](https://github.com/subukata/ubukit/blob/35389f9d9a101c5a8383d50e6a659fbed5b7543e/docs/numerics.md#L83-L91)

The recorded Optuna 5.0.0 comparison is exploratory: four handpicked cheap
problems, five seeds and 50 objective evaluations. UbuKit startup is 12 successful
observations versus Optuna's default 10, and equal integer seeds do not produce
matching initial samples across different RNGs. Joint UbuKit wins three of the
four medians against default Optuna, but independent UbuKit is worse than random
on the mixed problem. This is not held-out evidence, a significance result, a
state-of-the-art claim, a real clustering/SOM tuning evaluation, or a timing claim.

## JavaScript external-metric reference measurements

Source: [docs/numerics.md:118–123](https://github.com/subukata/ubukit/blob/35389f9d9a101c5a8383d50e6a659fbed5b7543e/docs/numerics.md#L118-L123)

The independent 80-digit oracle panel measured maximum AMI absolute error
3.33e-16 and exact final-Number ARI agreement. The benchmark results are bounded,
machine-specific measurements; see [the report](https://github.com/subukata/ubukit/blob/5a197111b605c495f3f2a7ba69a6e09bb44e8caf/preview/javascript/REPORT.md). Node/VM
smoke checks do not establish actual browser execution or cross-platform parity.

## Python exceptional-path timing observation

Source: [docs/numerics.md:153–154](https://github.com/subukata/ubukit/blob/35389f9d9a101c5a8383d50e6a659fbed5b7543e/docs/numerics.md#L153-L154)

Extreme correctness
recovery can be slower, including about 5.1x in one Python high-m warm smoke case.

## Python optimizer smoke comparisons and timing caveat

Source: [python/OPTIMIZATION.md:194–208](https://github.com/subukata/ubukit/blob/35389f9d9a101c5a8383d50e6a659fbed5b7543e/python/OPTIMIZATION.md#L194-L208)

## Evidence and validation

This preview has unit/lifecycle/numerical tests and shared Python/JS parity
fixtures. A small smoke suite uses four cheap problems, five seeds and fifty
objective evaluations per run. It compares random, independent TPE, joint TPE,
Optuna 5.0 default TPE, and Optuna 5.0 with `multivariate=False`.

In that particular smoke, joint TPE's median was better than random on all four
problems and better than Optuna 5.0 default on three; Optuna was better on
Branin. Independent UbuKit TPE was worse than random on the mixed problem.
This tiny handpicked suite is not a held-out benchmark or statistical proof of
superiority. Timing recorded while other processes ran is exploratory only.
Full per-seed values and settings are retained in the integration reports.

## Complete API-check example

The original assertion-based Python example remains in
[examples/python/all_methods.py](../examples/python/all_methods.py), unchanged.
Its prior README presentation is preserved in the
[original Python README](https://github.com/subukata/ubukit/blob/35389f9d9a101c5a8383d50e6a659fbed5b7543e/python/README.md#core-clustering-and-metric-examples).
The user guide now shows short examples without test-status output.

## Earlier research reports

The [research archive](performance/README.md) retains the original efficiency
and parameter-search reports. These records do not qualify a new build.
