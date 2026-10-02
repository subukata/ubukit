# Private combined preview: numerical contracts and limits

## FCM

The ordinary finite-m path remains in use when its range guards pass. Exceptional
near-one or large m, unsafe distances and unsafe weighted products use scaled
finite-m log-weight arithmetic. Positive distances are not clipped to zero, m is
not clipped, and no m-to-infinity hard or uniform replacement is substituted.
Distance-derived center weights are retained independently of rounded public
memberships. A new exceptional stopping-norm guard preserves representable tiny
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

Near-coincident centers are ill-conditioned: the existing m=64, seed994 fixture
changes memberships by as much as approximately 0.6505 versus dev2 even though
center changes are small. An independent 400-digit six-step trajectory improves
center accuracy in that case, but membership accuracy is not uniformly improved.
The detailed report and full reference evidence are retained. No arbitrary
precision, globally closer trajectory, global optimality, or cross-platform bit
identity is promised. Reinitializing from rounded public U cannot restore hidden
weights from an earlier run.

## SOM

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

A narrow prototype-mean repair can also activate after ordinary inputs generate
tiny unit masses. One inherited JavaScript case generates mass around 1e-117:
its repaired W[5] is 0.6884290744771887, exactly the rounded independent 160-digit
weighted-mean reference, rather than the old 0.6884290744771885. A second inherited case generates a tiny unit mass that rounds to zero after
the update; its W[24] changes from 2.9543435536324973 to 2.954343553632498.
These two coordinates are tested against separately audited exact weighted-mean
references. All other entries, events, P/V/history, labels and iteration results
remain exact; the other 78 cases retain their original exact comparison.

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

The recorded Optuna 5.0.0 comparison is exploratory: four handpicked cheap
problems, five seeds and 50 objective evaluations. UbuKit startup is 12 successful
observations versus Optuna's default 10, and equal integer seeds do not produce
matching initial samples across different RNGs. Joint UbuKit wins three of the
four medians against default Optuna, but independent UbuKit is worse than random
on the mixed problem. This is not held-out evidence, a significance result, a
state-of-the-art claim, a real clustering/SOM tuning evaluation, or a timing claim.

## Coverage boundary

Validation is Linux x86-64, CPython 3.12 and Node. Other operating systems,
architectures, Python versions and real browsers are unverified. Local bounded
benchmarks do not establish universal performance bounds. Extreme correctness
recovery can be slower, including about 5.1x in one Python high-m warm smoke case.
No public push, package-registry publication, deployment or upload was performed
while constructing this local bundle.
