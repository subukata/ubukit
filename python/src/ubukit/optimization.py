"""Dependency-free fixed-space hyperparameter optimization.

Multivariate (mixture of product kernels) TPE, optional independent TPE, and
random search. No Optuna compatibility or global-optimum guarantee is implied.
The optimizer owns one in-memory state; ask/tell can drive external workers but
is not a thread-safe distributed store. See OPTIMIZATION.md for numerical,
reproducibility, noise, and comparison contracts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real
from operator import index
from typing import Any, Callable, Mapping

__all__ = ["TPEOptimizer", "optimize", "float_range", "int_range", "categorical",
           "Trial", "OptimizationResult", "SearchSpaceExhausted", "ProposalError"]
_VERSION = "ubukit-tpe-1"
_MAX_SAFE = 9007199254740991
_SQRT2PI = 2.5066282746310002
_NEG_INF = float("-inf")


def _integer(value, name, low=0, high=_MAX_SAFE):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    try:
        value = index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]") from exc
    if not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _boolean(value, name):
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _scalar_key(value, *, categorical=False):
    if value is None:
        return ("null",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, str):
        return ("string", value)
    if categorical and isinstance(value, int) and not -_MAX_SAFE <= value <= _MAX_SAFE:
        raise ValueError("categorical integers must be within the JavaScript safe-integer range")
    number = _number(value, "categorical choice" if categorical else "parameter value")
    if categorical and number.is_integer() and abs(number) > _MAX_SAFE:
        raise ValueError("categorical integers must be within the JavaScript safe-integer range")
    return ("number", number)


class _RNG:
    """Mulberry32: all integer steps match JS Math.imul / >>> exactly."""
    def __init__(self, seed):
        self.state = _integer(seed, "seed", 0, 0xFFFFFFFF)

    def uint32(self):
        self.state = (self.state + 0x6D2B79F5) & 0xFFFFFFFF
        t = ((self.state ^ (self.state >> 15)) * (1 | self.state)) & 0xFFFFFFFF
        t ^= (t + (((t ^ (t >> 7)) * (61 | t)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return (t ^ (t >> 14)) & 0xFFFFFFFF

    def random(self):
        return self.uint32() / 4294967296.0

    def open(self):
        return (self.uint32() + 0.5) / 4294967296.0


def _normal_sf_positive(z):
    # Abramowitz-Stegun 26.2.17. Same coefficients/order in Python and JS.
    # Absolute CDF error <= ~7.5e-8; this is not a correctly-rounded libm CDF.
    if z == 0.0:
        return 0.5
    t = 1.0 / (1.0 + 0.2316419 * z)
    p = t * (0.319381530 + t * (-0.356563782 + t * (
        1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    return math.exp(-0.5 * z * z) * p / _SQRT2PI


def _normal_cdf(x):
    return _normal_sf_positive(-x) if x < 0.0 else 1.0 - _normal_sf_positive(x)


def _normal_small_logmass(midpoint, width):
    """Stable centered normal-bin mass with a bounded Taylor remainder.

    width is the full standardized interval width. For width <= .01 and
    abs(midpoint)*width <= .1, the degree-four integrated Taylor expansion has
    relative remainder < 4e-12. This follows by bounding the sixth derivative
    of the normal density on the interval; see OPTIMIZATION.md. Keeping width
    separate avoids cancellation when the rounded endpoints coincide.
    """
    product = midpoint*width
    if width > 0.01 or abs(product) > 0.1:
        return None
    w2 = width*width
    p2 = product*product
    correction = (p2-w2)/24.0 + (p2*p2-6.0*p2*w2+3.0*w2*w2)/1920.0
    return (-0.5*midpoint*midpoint-math.log(_SQRT2PI)+math.log(width)
            + math.log1p(correction))


def _normal_interval(a, b):
    # Tail symmetry avoids subtracting two numbers rounded to one.
    if b <= a:
        return 0.0
    narrow = _normal_small_logmass(a*0.5+b*0.5, b-a)
    if narrow is not None:
        return math.exp(narrow)
    if a >= 0.0:
        value = _normal_sf_positive(a) - _normal_sf_positive(b)
    elif b <= 0.0:
        value = _normal_sf_positive(-b) - _normal_sf_positive(-a)
    else:
        value = 1.0 - _normal_sf_positive(-a) - _normal_sf_positive(b)
    return max(0.0, value)


def _normal_ppf(p):
    # Peter J. Acklam's inverse-normal rational approximation coefficients.
    # We keep the same expression order in both languages; no libm erf needed.
    p = min(1.0 - 1e-15, max(1e-15, p))
    a = (-39.69683028665376, 220.9460984245205, -275.9285104469687,
         138.3577518672690, -30.66479806614716, 2.506628277459239)
    b = (-54.47609879822406, 161.5858368580409, -155.6989798598866,
         66.80131188771972, -13.28068155288572)
    c = (-0.007784894002430293, -0.3223964580411365, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783)
    d = (0.007784695709041462, 0.3224671290700398, 2.445134137142996,
         3.754408661907416)
    if p < 0.02425 or p > 0.97575:
        q = math.sqrt(-2.0 * math.log(p if p < 0.02425 else 1.0 - p))
        z = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / (
            (((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
        return z if p < 0.02425 else -z
    q = p - 0.5
    r = q*q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (
        (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r)+1.0)


def _logsumexp(values):
    maximum = max(values)
    if maximum == _NEG_INF:
        return maximum
    total = 0.0
    for value in values:
        total += math.exp(value - maximum)
    return maximum + math.log(total)


def float_range(low, high, *, log=False):
    """Bounded continuous range; log=True uses a log-uniform prior."""
    spec = {"type": "float", "low": low, "high": high, "log": log}
    _Domain("parameter", spec)
    return spec


def int_range(low, high, *, log=False):
    """Inclusive integer range. Log integer prior uses transformed bin mass."""
    spec = {"type": "int", "low": low, "high": high, "log": log}
    _Domain("parameter", spec)
    return spec


def categorical(choices):
    """Unique finite JSON scalars; numeric integers must be JavaScript-safe."""
    spec = {"type": "categorical", "choices": list(choices)}
    _Domain("parameter", spec)
    return spec


class _Domain:
    def __init__(self, name, spec):
        if not isinstance(spec, Mapping):
            raise ValueError(f"{name}: range must be a mapping")
        self.name = name
        self.kind = spec.get("type")
        if self.kind == "categorical":
            if set(spec) - {"type", "choices"}:
                raise ValueError(f"{name}: unsupported categorical range fields")
            values = spec.get("choices")
            if not isinstance(values, (list, tuple)) or not values:
                raise ValueError(f"{name}: choices must be a nonempty list")
            keys = [_scalar_key(value, categorical=True) for value in values]
            if len(set(keys)) != len(keys):
                raise ValueError(f"{name}: duplicate categorical choices")
            self.choices = tuple(values)
            self.lookup = {key: i for i, key in enumerate(keys)}
            self.size = len(values)
            self.fixed = self.size == 1
            self.log = False
            self.spec = {"type": self.kind, "choices": list(values)}
            return
        if self.kind not in ("int", "float"):
            raise ValueError(f"{name}: type must be float, int, or categorical")
        if set(spec) - {"type", "low", "high", "log"}:
            raise ValueError(f"{name}: unsupported numeric range fields")
        self.log = _boolean(spec.get("log", False), f"{name}.log")
        if self.kind == "int":
            self.low = _integer(spec.get("low"), f"{name}.low", -_MAX_SAFE)
            self.high = _integer(spec.get("high"), f"{name}.high", -_MAX_SAFE)
        else:
            self.low = _number(spec.get("low"), f"{name}.low")
            self.high = _number(spec.get("high"), f"{name}.high")
        if self.low > self.high or (self.log and self.low <= 0):
            raise ValueError(f"{name}: require low <= high and positive log bounds")
        self.fixed = self.low == self.high
        self.size = self.high - self.low + 1 if self.kind == "int" else (1 if self.fixed else None)
        if self.kind == "int" and self.size > _MAX_SAFE:
            raise ValueError(f"{name}: integer cardinality exceeds JS safe integer range")
        self.span = 1.0
        self.close_log = False
        if self.log and self.kind == "int":
            self.base = self.low - 0.5
            self.span = math.log1p(self.size / self.base)
        elif self.log and not self.fixed:
            self.log_low = math.log(self.low)
            ratio = (self.high - self.low) / self.low
            self.close_log = math.isfinite(ratio) and ratio <= 1.0
            self.span = math.log1p(ratio) if self.close_log else math.log(self.high)-self.log_low
        elif self.kind == "float" and not self.fixed:
            self.span = self.high - self.low
        self.spec = {"type": self.kind, "low": self.low, "high": self.high, "log": self.log}

    def encode(self, value):
        if self.kind == "categorical":
            try:
                return self.lookup[_scalar_key(value, categorical=True)]
            except KeyError as exc:
                raise ValueError(f"{self.name}: value is not in choices") from exc
        value = (_integer(value, self.name, -_MAX_SAFE) if self.kind == "int"
                 else _number(value, self.name))
        if not self.low <= value <= self.high:
            raise ValueError(f"{self.name}: value outside search range")
        if self.fixed:
            return 0.5
        if self.kind == "int":
            offset = value - self.low + 0.5
            return math.log1p(offset / self.base)/self.span if self.log else offset/self.size
        if self.log:
            return (math.log1p((value-self.low)/self.low) if self.close_log
                    else math.log(value)-self.log_low)/self.span
        if math.isfinite(self.span):
            return (value-self.low)/self.span
        return (value*0.5-self.low*0.5)/(self.high*0.5-self.low*0.5)

    def decode(self, value):
        if self.kind == "categorical":
            return self.choices[int(value)]
        if self.fixed:
            return self.low
        value = min(1.0, max(0.0, value))
        if value == 0.0:
            return self.low
        if value == 1.0:
            return self.high
        if self.kind == "int":
            offset = self.base*math.expm1(value*self.span) if self.log else value*self.size
            return self.low + min(self.size-1, max(0, math.floor(offset)))
        if self.log:
            raw = (self.low+self.low*math.expm1(value*self.span) if self.close_log
                   else math.exp(self.log_low+value*self.span))
        else:
            raw = (1.0-value)*self.low + value*self.high
        return min(self.high, max(self.low, raw))

    def bins(self, value):
        offset = value-self.low
        if self.log:
            return (math.log1p(offset/self.base)/self.span,
                    math.log1p((offset+1)/self.base)/self.span)
        return offset/self.size, (offset+1)/self.size

    def bin_width(self, value):
        # b-a loses all digits for narrow log-integer bins near the upper edge.
        return (math.log1p(1.0/(self.base+value-self.low))/self.span
                if self.log else 1.0/self.size)

    def random(self, rng):
        if self.kind == "categorical":
            return self.decode(min(self.size-1, int(rng.random()*self.size)))
        return self.decode(rng.open())

    def prior_log(self, value):
        if self.fixed:
            return 0.0
        if self.kind == "categorical":
            return -math.log(self.size)
        if self.kind == "int":
            return math.log(self.bin_width(value))
        return 0.0  # normalized latent space; Jacobian cancels in l/g

    def discrete_value(self, offset):
        if self.kind == "categorical":
            return self.choices[offset]
        return self.low + offset if self.kind == "int" else self.low


class _KDE:
    """Mixture of independent per-axis kernels; shared mixture index is joint."""
    def __init__(self, domains, rows, weights, multivariate, min_bandwidth):
        self.domains = domains
        self.rows = rows
        self.n = len(rows)
        self.multivariate = multivariate
        total = sum(weights) + 1.0
        self.weights = [w/total for w in weights] + [1.0/total]
        self.log_weights = [math.log(w) if w > 0 else _NEG_INF for w in self.weights]
        self.centers = [[d.encode(row[d.name]) for row in rows] for d in domains]
        self.sigmas, self.norms, self.cdf_lows = [], [], []
        self.cat_alpha = 1.0/(self.n+1.0)
        floor = max(min_bandwidth, 1.0/(self.n+1.0)**2)
        for d, centers in zip(domains, self.centers):
            if d.kind == "categorical" or d.fixed:
                self.sigmas.append(None)
                self.norms.append(None)
                self.cdf_lows.append(None)
                continue
            # Include uniform prior's midpoint as a bandwidth neighbor only.
            ordered = sorted([(x, i) for i, x in enumerate(centers)] + [(0.5, self.n)])
            sigmas = [0.0]*self.n
            for pos, (x, i) in enumerate(ordered):
                if i == self.n:
                    continue
                left = ordered[pos-1][0] if pos else 0.0
                right = ordered[pos+1][0] if pos+1 < len(ordered) else 1.0
                sigmas[i] = min(1.0, max(floor, x-left, right-x))
            lows = [_normal_cdf(-mu/s) for mu, s in zip(centers, sigmas)]
            norms = [_normal_interval(-mu/s, (1.0-mu)/s) for mu, s in zip(centers, sigmas)]
            self.sigmas.append(sigmas)
            self.cdf_lows.append(lows)
            self.norms.append(norms)

    def _component(self, rng):
        u = rng.random()
        total = 0.0
        for i, w in enumerate(self.weights):
            total += w
            if u < total:
                return i
        return self.n

    def sample(self, rng):
        component = self._component(rng) if self.multivariate else None
        out = {}
        for j, d in enumerate(self.domains):
            i = component if self.multivariate else self._component(rng)
            if i == self.n:
                out[d.name] = d.random(rng)
            elif d.kind == "categorical":
                # Aitchison-Aitken equivalent: center spike + uniform smoothing.
                u = rng.random()
                alpha = self.cat_alpha
                if u < 1.0-alpha:
                    k = self.centers[j][i]
                else:
                    k = min(d.size-1, math.floor((u-(1.0-alpha))/alpha*d.size))
                out[d.name] = d.decode(k)
            elif d.fixed:
                rng.open()  # specified draw count independent of fixedness
                out[d.name] = d.low
            else:
                p = self.cdf_lows[j][i] + rng.open()*self.norms[j][i]
                z = self.centers[j][i] + self.sigmas[j][i]*_normal_ppf(p)
                out[d.name] = d.decode(z)
        return out

    def _axis_logs(self, j, value):
        d = self.domains[j]
        if d.fixed:
            return [0.0]*(self.n+1)
        if d.kind == "categorical":
            encoded = d.encode(value)
            off = self.cat_alpha/d.size
            on = 1.0-self.cat_alpha+off
            return [math.log(on if x == encoded else off) for x in self.centers[j]] + [-math.log(d.size)]
        if d.kind == "int":
            low, high = d.bins(value)
            width = d.bin_width(value)
            logs = []
            for mu, sigma, norm in zip(self.centers[j], self.sigmas[j], self.norms[j]):
                midpoint = ((low+high)*0.5-mu)/sigma
                narrow = _normal_small_logmass(midpoint, width/sigma)
                if narrow is not None:
                    logs.append(narrow-math.log(norm))
                    continue
                mass = _normal_interval((low-mu)/sigma, (high-mu)/sigma)
                logs.append(math.log(mass/norm) if mass > 0 else _NEG_INF)
            return logs + [d.prior_log(value)]
        x = d.encode(value)
        return [-0.5*((x-mu)/sigma)**2-math.log(sigma*_SQRT2PI*norm)
                for mu, sigma, norm in zip(self.centers[j], self.sigmas[j], self.norms[j])] + [0.0]

    def logpdf(self, params):
        if self.multivariate:
            components = self.log_weights.copy()
            for j, d in enumerate(self.domains):
                logs = self._axis_logs(j, params[d.name])
                for i in range(self.n+1):
                    components[i] += logs[i]
            return _logsumexp(components)
        value = 0.0
        for j, d in enumerate(self.domains):
            logs = self._axis_logs(j, params[d.name])
            value += _logsumexp([a+b for a, b in zip(self.log_weights, logs)])
        return value


class SearchSpaceExhausted(RuntimeError):
    """All settings in an exactly counted finite space have been reserved."""


class ProposalError(RuntimeError):
    """Bounded retries failed; does not claim the search space is exhausted."""


@dataclass(frozen=True)
class Trial:
    id: int
    params: dict
    state: str = "running"
    value: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class OptimizationResult:
    best_params: dict | None
    best_value: float | None
    best_trial_id: int | None
    history: list[Trial]
    n_attempted: int
    n_completed: int
    stop_reason: str

    @property
    def trials(self):
        return self.history

    def to_dict(self):
        return asdict(self)


class TPEOptimizer:
    """Small single-objective ask/tell optimizer with JSON-compatible ranges.

    ``multivariate=True`` shares each mixture component across dimensions;
    it captures dependencies through joint observed configurations without a
    full covariance matrix. ``False`` provides an independent-TPE ablation.
    The default is a research-informed heuristic, not an Optuna reproduction.
    """
    def __init__(self, space, *, seed=0, direction="minimize", sampler="tpe",
                 n_startup_trials=12, n_candidates=24, gamma=0.15,
                 multivariate=True, min_bandwidth=0.03, weights="ei",
                 avoid_duplicates=True):
        if not isinstance(space, Mapping) or not space:
            raise ValueError("space must be a nonempty mapping of names to ranges")
        if any(not isinstance(k, str) or not k for k in space):
            raise ValueError("parameter names must be nonempty strings")
        self._domains = [_Domain(k, space[k]) for k in sorted(space, key=lambda x: x.encode("utf-16-be", "surrogatepass"))]
        if direction not in ("minimize", "maximize") or sampler not in ("tpe", "random"):
            raise ValueError("direction must be minimize/maximize; sampler must be tpe/random")
        self.direction, self.sampler = direction, sampler
        self.n_startup_trials = _integer(n_startup_trials, "n_startup_trials", 2)
        self.n_candidates = _integer(n_candidates, "n_candidates", 1)
        self.gamma = _number(gamma, "gamma")
        self.min_bandwidth = _number(min_bandwidth, "min_bandwidth")
        if not 0.0 < self.gamma < 1.0 or not 0.0 < self.min_bandwidth <= 1.0:
            raise ValueError("require 0 < gamma < 1 and 0 < min_bandwidth <= 1")
        self.multivariate = _boolean(multivariate, "multivariate")
        self.avoid_duplicates = _boolean(avoid_duplicates, "avoid_duplicates")
        if weights not in ("ei", "uniform"):
            raise ValueError("weights must be ei or uniform")
        self.weighting = weights
        self._rng = _RNG(seed)
        self.seed = seed
        self._trials = []
        self._seen = set()
        self._completed = []
        self._models = None
        self._finite_size = 1
        for d in self._domains:
            if d.size is None:
                self._finite_size = None
                break
            self._finite_size *= d.size
        self.stop_reason = "not_started"

    def _key(self, params):
        return tuple(_scalar_key(params[d.name]) for d in self._domains)

    def _random(self):
        return {d.name: d.random(self._rng) for d in self._domains}

    def _available(self, params):
        return not self.avoid_duplicates or self._key(params) not in self._seen

    def _fit(self):
        if self._models is not None:
            return self._models
        sign = 1.0 if self.direction == "minimize" else -1.0
        ordered = sorted(self._completed, key=lambda t: (sign*t.value, t.id))
        n = len(ordered)
        split = min(n-1, max(1, math.ceil(self.gamma*n)))
        good, bad = ordered[:split], ordered[split:]
        weights = [1.0]*split
        if self.weighting == "ei":
            # Normalize before subtracting to avoid overflow of finite losses.
            scale = max(abs(t.value) for t in ordered) or 1.0
            threshold = sign*bad[0].value/scale
            diffs = [max(0.0, threshold-sign*t.value/scale) for t in good]
            total = sum(diffs)
            if total > 0.0:
                weights = [max(1e-12, v/total*split) for v in diffs]
                mean = sum(weights)/split
                weights = [w/mean for w in weights]
        self._models = (
            _KDE(self._domains, [t.params for t in good], weights, self.multivariate, self.min_bandwidth),
            _KDE(self._domains, [t.params for t in bad], [1.0]*len(bad), self.multivariate, self.min_bandwidth))
        return self._models

    def _propose(self):
        if self.avoid_duplicates and self._finite_size is not None and len(self._seen) >= self._finite_size:
            raise SearchSpaceExhausted("all settings in the finite search space have been issued")
        model_ready = len(self._completed) >= self.n_startup_trials
        if model_ready and self.sampler == "tpe":
            # With no signal, do not impose arbitrary tied ranks on KDEs.
            first = self._completed[0].value
            if any(t.value != first for t in self._completed[1:]):
                good, bad = self._fit()
                best, best_score = None, _NEG_INF
                for _ in range(self.n_candidates):
                    params = good.sample(self._rng)
                    if not self._available(params):
                        continue
                    score = good.logpdf(params)-bad.logpdf(params)
                    if best is None or score > best_score:
                        best, best_score = params, score
                if best is not None:
                    return best
        for _ in range(64):
            params = self._random()
            if self._available(params):
                return params
        # Exact fallback only for small finite spaces; bounded runtime otherwise.
        if self._finite_size is not None and self._finite_size <= 100000:
            start = min(self._finite_size-1, int(self._rng.random()*self._finite_size))
            for shift in range(self._finite_size):
                code = (start+shift) % self._finite_size
                params = {}
                for d in reversed(self._domains):
                    code, offset = divmod(code, d.size)
                    params[d.name] = d.discrete_value(offset)
                params = {d.name: params[d.name] for d in self._domains}
                if self._available(params):
                    return params
        raise ProposalError("could not propose a new setting after bounded retries; allow duplicates or change the space")

    def ask(self):
        """Reserve a candidate; only completed successful trials train the model."""
        params = self._propose()
        trial = Trial(len(self._trials), params)
        self._trials.append(trial)
        self._seen.add(self._key(params))
        self.stop_reason = "running"
        return Trial(trial.id, params.copy())

    def tell(self, trial_id, value=None, *, state="complete", error=None):
        """Finish one issued ID exactly once; fail/cancel records have no value."""
        trial_id = _integer(trial_id, "trial_id")
        if trial_id >= len(self._trials) or self._trials[trial_id].state != "running":
            raise ValueError("tell requires a currently running trial ID issued by this optimizer")
        if state not in ("complete", "fail", "cancelled"):
            raise ValueError("state must be complete, fail, or cancelled")
        if error is not None and not isinstance(error, str):
            raise ValueError("error must be a string or None")
        if state == "complete":
            value = _number(value, "objective value")
            if error is not None:
                raise ValueError("a complete trial cannot contain an error")
        elif value is not None:
            raise ValueError("failed/cancelled trials cannot have an objective value")
        old = self._trials[trial_id]
        trial = Trial(trial_id, old.params.copy(), state, value, error)
        self._trials[trial_id] = trial
        if state == "complete":
            self._completed.append(trial)
            self._models = None
        return Trial(trial.id, trial.params.copy(), trial.state, trial.value, trial.error)

    def add_trial(self, params, value=None, *, state="complete", error=None):
        """Import a known observation without drawing RNG; duplicates are real data.

        Space and finite-value validation are identical to generated trials.
        Unlike tell(), this allocates an ID for an already evaluated setting.
        """
        if not isinstance(params, Mapping) or set(params) != {d.name for d in self._domains}:
            raise ValueError("params must contain exactly the search-space parameter names")
        canonical = {}
        for d in self._domains:
            encoded = d.encode(params[d.name])
            # Do not re-round continuous imported values through log/exp.
            canonical[d.name] = (d.choices[encoded] if d.kind == "categorical" else
                                 int(params[d.name]) if d.kind == "int" else float(params[d.name]))
        if state not in ("complete", "fail", "cancelled"):
            raise ValueError("state must be complete, fail, or cancelled")
        if error is not None and not isinstance(error, str):
            raise ValueError("error must be a string or None")
        if state == "complete":
            value = _number(value, "objective value")
            if error is not None:
                raise ValueError("a complete trial cannot contain an error")
        elif value is not None:
            raise ValueError("failed/cancelled trials cannot have an objective value")
        trial = Trial(len(self._trials), canonical, state, value, error)
        self._trials.append(trial)
        self._seen.add(self._key(canonical))
        if state == "complete":
            self._completed.append(trial)
            self._models = None
        return Trial(trial.id, canonical.copy(), state, value, error)

    def result(self):
        """Snapshot; best means best observed successful objective, not true optimum."""
        sign = 1.0 if self.direction == "minimize" else -1.0
        best = min(self._completed, key=lambda t: (sign*t.value, t.id), default=None)
        history = [Trial(t.id, t.params.copy(), t.state, t.value, t.error) for t in self._trials]
        return OptimizationResult(best.params.copy() if best else None, best.value if best else None,
                                  best.id if best else None, history, len(history), len(self._completed), self.stop_reason)


def optimize(objective: Callable[[dict], float], space, n_trials=None, *, budget=None,
             continue_on_error=False, callback=None, **options):
    """Evaluate at most n_trials settings sequentially; budget is an alias.

    Exceptions and nonfinite returns record FAIL and are re-raised by default.
    With continue_on_error=True, ordinary Exception failures consume one trial
    and evaluation continues. KeyboardInterrupt/SystemExit are never swallowed.
    Callback receives a detached finished Trial; return False to stop.
    """
    if not callable(objective):
        raise ValueError("objective must be callable")
    if n_trials is not None and budget is not None:
        raise ValueError("specify n_trials or budget, not both")
    count = _integer(budget if n_trials is None else n_trials, "n_trials")
    _boolean(continue_on_error, "continue_on_error")
    if callback is not None and not callable(callback):
        raise ValueError("callback must be callable")
    search = TPEOptimizer(space, **options)
    search.stop_reason = "budget_exhausted"
    for _ in range(count):
        try:
            trial = search.ask()
        except SearchSpaceExhausted:
            search.stop_reason = "space_exhausted"
            break
        except ProposalError:
            search.stop_reason = "proposal_failed"
            break
        try:
            value = objective(trial.params.copy())
            finished = search.tell(trial.id, value)
        except Exception as exc:
            finished = search.tell(trial.id, state="fail", error=f"{type(exc).__name__}: {exc}")
            if not continue_on_error:
                raise
        except BaseException:
            search.tell(trial.id, state="cancelled")
            raise
        if callback is not None and callback(finished) is False:
            search.stop_reason = "callback_stopped"
            break
        search.stop_reason = "budget_exhausted"
    return search.result()
