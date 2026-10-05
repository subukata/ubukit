"""Time and score the same cases in Python and JavaScript.

Run from the repository root with Node.js on PATH:

    python bench/run.py                # full sizes, best of 3 runs
    python bench/run.py --only fcm,ami # a subset
    python bench/run.py --smoke        # tiny sizes; CI uses this to keep the script working

Paste the table, from before and after, into pull requests that change an
algorithm. Cases are defined only here; bench/run.mjs runs the same list in
JavaScript. Quality is computed here for both languages: ARI against the
generating labels for clustering, mean squared quantization error for maps,
and the returned value for metrics and search.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.spatial.distance import cdist

import ubukit as ub

HERE = Path(__file__).resolve().parent
SIZES = {
    "full": {"n": 20_000, "small": 5_000, "embed": 3_000, "labels": 1_000_000},
    "smoke": {"n": 600, "small": 200, "embed": 100, "labels": 10_000},
}
# (name, method, data, positional args, options, quality); strings in args and
# options that name a dataset are replaced by it.
CASES = [
    ("kmeans", "kmeans", "X", [10], {"init": "init"}, "ari"),
    ("fcm", "fcm", "X", [10], {"init": "init"}, "ari"),
    ("efcm", "efcm", "X", [10], {"init": "init", "tau": 5.0}, "ari"),
    ("rcm", "rcm", "X", [10], {"init": "init", "alpha": 1.2}, "ari"),
    ("rmcm", "rmcm", "small", [10, 3.0], {"init": "init"}, "ari"),
    ("batch_som", "batch_som", "X", [[10, 10]], {"epochs": 20}, "qe"),
    ("som", "som", "small", [[10, 10]], {"epochs": 2, "seed": 0}, "qe"),
    ("som_olp", "som_olp", "small", [[10, 10]], {"lam": 5.0, "gamma": 1.0}, "qe"),
    ("trustworthiness", "trustworthiness", "embed", ["embed_y", 10], {}, "value"),
    ("ami", "ami", "labels_a", ["labels_b"], {}, "value"),
    ("tpe", "tpe_sphere", None, [200], {"seed": 0}, "best"),
]
TRUTH = {"X": "y", "small": "y_small"}


def datasets(size: dict) -> dict:
    """Seeded Gaussian blobs (n x 16 around 10 centers) and the sets derived from them."""
    rng = np.random.default_rng(0)
    centers = rng.uniform(-5, 5, (10, 16))
    y = rng.permutation(np.arange(size["n"]) % 10)
    X = centers[y] + rng.normal(size=(size["n"], 16))
    embed = X[: size["embed"]]
    return {
        "X": X,
        "y": y,
        "init": X[rng.choice(size["n"], 10, replace=False)] + 0.01,
        "small": X[: size["small"]],
        "y_small": y[: size["small"]],
        "embed": embed,
        "embed_y": embed[:, :2] + 0.3 * rng.normal(size=(len(embed), 2)),
        "labels_a": rng.integers(0, 50, size["labels"]),
        "labels_b": rng.integers(0, 40, size["labels"]),
    }


def tpe_sphere(n_trials, *, seed):
    space = {"x": ub.uniform(-5, 5), "y": ub.uniform(-5, 5)}
    return ub.minimize(lambda p: (p["x"] - 1) ** 2 + p["y"] ** 2, space, n_trials, seed=seed)


def run_python(data: dict, cases: list, repeat: int) -> dict:
    out = {}
    for name, method, X, args, options, _ in cases:
        fn = tpe_sphere if method == "tpe_sphere" else getattr(ub, method)
        call_args = ([] if X is None else [data[X]]) + [_resolve(a, data) for a in args]
        call_options = {k: _resolve(v, data) for k, v in options.items()}
        best = np.inf
        for _ in range(repeat):
            start = time.perf_counter()
            result = fn(*call_args, **call_options)
            best = min(best, time.perf_counter() - start)
        out[name] = {"seconds": best, **_summary(result)}
    return out


def run_javascript(data: dict, cases: list, repeat: int) -> dict:
    payload = {
        "data": {k: v.tolist() for k, v in data.items()},
        "cases": [
            {"name": n, "method": m, "X": X, "args": a, "options": o} for n, m, X, a, o, _ in cases
        ],
        "repeat": repeat,
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bench.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        done = subprocess.run(
            ["node", str(HERE / "run.mjs"), str(path)], capture_output=True, text=True, check=True
        )
    return json.loads(done.stdout)


def quality(kind: str, result: dict, data: dict, X: str | None) -> str:
    if kind == "ari":
        return f"ARI {ub.ari(data[TRUTH[X]], result['labels']):.4f}"
    if kind == "qe":
        qe = cdist(data[X], np.asarray(result["centers"]), "sqeuclidean").min(axis=1).mean()
        return f"QE {qe:.4f}"
    return f"{result['value']:.6g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true", help="tiny sizes, one run")
    parser.add_argument("--repeat", type=int, help="runs per case (best is reported)")
    parser.add_argument("--only", help="comma-separated case names")
    opts = parser.parse_args()
    size = SIZES["smoke" if opts.smoke else "full"]
    repeat = opts.repeat or (1 if opts.smoke else 3)
    cases = [c for c in CASES if not opts.only or c[0] in opts.only.split(",")]
    data = datasets(size)
    py = run_python(data, cases, repeat)
    js = run_javascript(data, cases, repeat)
    node = subprocess.run(["node", "--version"], capture_output=True, text=True).stdout.strip()
    print(
        f"Python {platform.python_version()}, NumPy {np.__version__}, SciPy {scipy.__version__},"
        f" Node {node}, {platform.system()} {platform.machine()}; N = {size['n']}, best of {repeat}"
    )
    print()
    print("| case | Python s | JS s | Python quality | JS quality |")
    print("|---|---:|---:|---|---|")
    for name, _, X, _, _, kind in cases:
        p, j = py[name], js[name]
        print(
            f"| {name} | {p['seconds']:.3f} | {j['seconds']:.3f}"
            f" | {quality(kind, p, data, X)} | {quality(kind, j, data, X)} |"
        )


def _resolve(value, data):
    return data[value] if isinstance(value, str) and value in data else value


def _summary(result) -> dict:
    if isinstance(result, ub.Result):
        return {"labels": result.labels, "centers": result.centers}
    if isinstance(result, ub.TPEResult):
        return {"value": result.best_value}
    return {"value": float(result)}


if __name__ == "__main__":
    main()
