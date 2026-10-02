# Numerical contracts and limits

## FCM

The ordinary finite-m path remains in use when its range guards pass. Exceptional
near-one or large m, unsafe distances and unsafe weighted products use scaled
finite-m log-weight arithmetic. Positive distances are not clipped to zero, m is
not clipped, and no m-to-infinity hard or uniform replacement is substituted.
Distance-derived center weights are retained independently of rounded public
memberships. An exceptional stopping-norm guard preserves representable tiny
membership changes whose squared norm would otherwise underflow.

Membership convergence retains the public absolute Frobenius criterion. A
rounded membership tie can coexist with nonuniform hidden log weights and
continuing center movement; the extra diagnostics expose this distinction.
Objective values outside scalar float64 range are reported as 0/infinity with
an explicit objective status and logarithmic value rather than silently clipped.
The objective is evaluated at the published rounded memberships and centers.

Memberships may be evaluated in translated working coordinates. Restoring centers
to very large public coordinates can round away their offsets. Diagnostics
membership_frame='working_coordinates' and published_centers_rounded disclose this
case. Memberships are deliberately not recomputed solely to force agreement with
the restored centers. The working-center/public-center difference is a real
limitation, not a guarantee of returned-pair consistency.

Near-coincident centers are ill-conditioned: small center changes can cause
large membership changes. Stabilization does not guarantee a globally closer
arbitrary-precision trajectory, global optimality, or cross-platform bit
identity. Reinitializing from rounded public memberships cannot restore hidden
weights from an earlier run.

## SOM-OLP

The exceptional path retains the original gamma and lambda, caller-unit history,
and relative stopping criterion. Python represents costs using mantissa/exponent
pairs, subtracting minima before dividing by the original temperature. Cold
weighted means use exact binary product accumulation before float64 rounding.
JavaScript uses a deliberately narrower finite-composite-cost contract and reports
its actual fallback even when WASM was requested. Positive composite costs that
become zero or overflow in that JavaScript contract raise explicit errors.

Both languages retain the minimized row-objective identity and small entropy
tails where representable. Unrepresentable final histories or prototypes raise.
No lambda floor, hard assignment, uniform substitute, or claimed global optimum
is introduced. Probabilities, outputs, ordinary distance products and LAPACK PCA
remain float64; extreme nearly tied quantities can remain sensitive.

A prototype-mean repair can also activate when ordinary inputs produce tiny
unit masses, including masses that round to zero after an update.

Exceptional Python initial memberships must be nonnegative and row-stochastic
within 1e-8. This is a deliberate restriction of an invalid numerical input domain;
ordinary invalid-input acceptance is not broadly redesigned. Public APIs are the
supported dispatch surface. Private direct backend calls can bypass that dispatch.
Primary scratch budgets exclude documented runtime allocations, outputs, Python
integer/JavaScript BigInt temporaries and some library workspace.

## Lightweight optimization

The optional optimizer adds no mandatory dependency. Python's optimization facade
is lazy and its implementation uses the standard library; JavaScript adds no npm
runtime dependency. Existing clustering scientific dependencies are unchanged.
It supports flat, single-objective float/int/categorical/log search with bounded
TPE or random sampling and seeded ask/tell. It does not implement conditional
trees, pruning, full covariance, multiobjective search or parallel orchestration.

CDF/inverse-CDF formulas are approximations. A bounded fourth-order centered
integral repairs narrow bin probabilities; broad non-narrow far-tail relative
accuracy remains limited. Safe-integer rules prevent category aliasing across
Python and JavaScript. Replay requires the same observations and ask/tell order;
libm and browser differences preclude a universal bit-parity guarantee.

Small synthetic comparisons do not establish superiority over random search or
Optuna. Use held-out validation and account for objective-evaluation cost.

## JavaScript external clustering metrics

ARI, AMI and shared-contingency joint scores add no runtime dependency. The
inputs are equal-length Arrays or numeric TypedArrays, each containing only
strings or only safe integer Numbers. The full contract and error codes are in
[JavaScript external metrics guide](../javascript/EXTERNAL_METRICS.md).
Python has a separately documented [external-metric API](../python/README.md#ari-and-ami).

ARI uses exact integer combinatorics, with BigInt when necessary, before the
final binary64 quotient. AMI uses direct conditional entropies, a full-support
mode-centered hypergeometric recurrence and compensated summation. Naturally
underflowed probability weights and binary64 rounding remain possible. This is
not arbitrary precision or a universal absolute-error guarantee, and does not
reproduce scikit-learn's high-K rounding artifacts.

AMI defaults to arithmetic normalization. Geometric, min and max are optional.
All-singletons versus a nonidentical nonconstant partition has undefined 0/0
min-normalized AMI and raises AMI_SINGULAR_NORMALIZATION; arithmetic, geometric
and max return zero. The joint API also raises when its AMI is unsupported.

AMI permits at most 2**26 samples. ARI's length bound is 2**32 - 1, subject to
available memory. AMI maxExpectedTerms defaults to 10,000,000 grouped support
terms; exceeding it raises AMI_WORK_LIMIT. This bound is neither a wall-clock
nor a total-memory guarantee. Many distinct large margins can be expensive.
The APIs are synchronous; large calls belong in a separately managed worker.
No scheduler/worker integration is added by this metric implementation.

Node/VM smoke checks do not establish actual browser execution or
cross-platform parity.

## Online SOM and BatchSOM

These are separate algorithms from SOM-OLP. Online updates one sample at a time
in input order; batch freezes all BMUs and commits one weighted-mean epoch.
The batch algorithm has no learning-rate parameter. Both support arbitrary
feature dimension and default to a planar rectangular 16×16 lattice.

Exact computed distance ties select the lowest unit index; near ties can differ
between floating-point summation orders and amplify across later steps. PCA
bases in degenerate cases need not match between Python and JavaScript. The
shared explicit-prototype fixtures establish a bounded parity scope, not universal
bit-identical trajectories, convergence or a global optimum.

Python has an exact-binary cold repair for extreme BMU ordering and protected
weighted means. JavaScript deliberately rejects unrepresentable squared distances
(overflow or underflow/subnormal) with rescaling guidance. Both retain old units
for zero effective neighborhood mass, including positive-Gaussian underflow.
Exact-arithmetic cancellation recovery can be slower and allocate extra memory.

Snapshots expose committed prototypes; JavaScript full-data projection is explicit
before completion. Soft chunk budgets are not hard realtime deadlines. Primary
memory estimates exclude runtime/library workspace, caller copies and exceptional
integer temporaries. Read the complete [Python](../python/SOM.md) and
[JavaScript](../javascript/SOM.md) contracts.

## Coverage boundary

Validation is Linux x86-64, CPython 3.12 and Node. Other operating systems,
architectures, Python versions and real browsers are unverified. Local bounded
benchmarks do not establish universal performance bounds. Extreme numerical
recovery can be slower.
This is a pre-release preview, not a published package.
