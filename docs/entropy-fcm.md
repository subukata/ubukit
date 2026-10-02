# Entropy-regularized fuzzy c-means

Entropy-regularized fuzzy c-means (EFCM) groups samples while retaining a degree
of membership in every cluster. A positive temperature controls softness.

For samples `x_i`, centers `v_k`, memberships `u_ik`, and `tau > 0`, minimize

\[
J(U,V)=\sum_{i,k}u_{ik}\lVert x_i-v_k\rVert^2
       +\tau\sum_{i,k}u_{ik}\log u_{ik},\qquad
u_{ik}\ge0,\quad\sum_k u_{ik}=1.
\]

The convention is `0 log 0 = 0`. The center and membership updates are

\[
v_k=\frac{\sum_i u_{ik}x_i}{\sum_i u_{ik}},\qquad
u_{ik}=\frac{\exp[-(d_{ik}^2-m_i)/\tau]}
                  {\sum_j\exp[-(d_{ij}^2-m_i)/\tau]},\quad
m_i=\min_j d_{ij}^2.
\]

Subtracting the minimum before dividing by `tau` avoids overflowing the
softmax normalization. A zero-distance center does not force a hard assignment
when `tau` is positive. Centers use linear membership weights; the classical
FCM fuzzifier `m` does not apply. Empty clusters retain their previous center,
initially the data mean.

Python: `ubukit.fit_entropy_fcm(X, 2, tau=0.5, random_state=4)`.
JavaScript: `entropyFcm(X, { nClusters: 2, tau: 0.5, seed: 4 })`, or
`run('entropy-fcm', X, options)`. See the language-specific
[Python options and results](../python/README.md#entropy-regularized-fuzzy-c-means)
and [JavaScript guide](../javascript/ENTROPY_FCM.md).

`tau` has the units of squared distance. If coordinates are multiplied by `s`,
multiply `tau` by `s²` to preserve the mathematical memberships. Standardizing
features changes the distance metric and should be a deliberate modeling choice.
For fixed centers, smaller temperatures approach nearest-center assignments
with equal sharing of exact ties; larger temperatures approach uniform membership.
Finite float64 results can underflow tiny memberships to zero. Extreme input
ranges use range-preserving arithmetic and explicit objective diagnostics.

Each iteration updates centers from old memberships, then memberships from those
centers. The reported objective evaluates the returned pair and can be negative.
The membership-change stopping rule does not prove a stationary center update or
a global optimum. Results from different seeds can differ; Python and JavaScript
use their existing, different random generators. Use explicit initial memberships
for cross-language comparisons.

EFCM and SOM-OLP share entropy regularization and a softmax assignment step, but
their losses, variables, and center/map updates differ. UbuKit shares numerical
arithmetic where appropriate; it does not substitute SOM-OLP training for EFCM.

Background: Miyamoto, Umayahara and Mukaidono (1998),
[Fuzzy Classification Functions in the Methods of Fuzzy c-Means and Regularization by Entropy](https://doi.org/10.3156/jfuzzy.10.3_548).
