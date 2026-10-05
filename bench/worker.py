"""Run benchmark cases in one Python process; bench/run.py drives it.

Usage: python bench/worker.py request.json  (results as JSON on stdout)

The request names the source tree whose UbuKit to import, so --compare can
time another commit with this same harness, and the implementation:
"python" (UbuKit) or "scikit-learn" (references for the same quantities).
"""

from __future__ import annotations

import json
import platform
import sys
import time
import tracemalloc
from pathlib import Path


def main(request_path: str) -> None:
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    sys.path.insert(0, str(Path(request["src"]) / "python" / "src"))
    import numpy as np
    import scipy
    import threadpoolctl

    import ubukit as ub

    functions = _references() if request["impl"] == "scikit-learn" else _ubukit(ub)
    data = _Data(Path(request["data"]), np)
    results = {}
    for case in request["cases"]:
        if case["method"] not in functions:
            continue
        fn = functions[case["method"]]
        args = [data.resolve(case["data"], a) for a in case["args"]]
        variants = [case["options"]]
        if request["seeds"] and case["seeded"]:
            fixed = {k: v for k, v in case["options"].items() if k not in ("init", "seed")}
            variants = [{**fixed, "seed": s} for s in request["seeds"]]
        try:
            entry = _run(fn, args, variants, data, case["data"], request["repeat"])
            if request["memory"]:
                tracemalloc.start()
                fn(*args, **_resolved(variants[0], data, case["data"]))
                entry["peak_mb"] = tracemalloc.get_traced_memory()[1] / 2**20
                tracemalloc.stop()
        except Exception as error:  # report, do not abort the other cases
            entry = {"error": f"{type(error).__name__}: {error}"}
        results[case["name"]] = entry
    pools = threadpoolctl.threadpool_info()
    blas = sorted({f"{i['internal_api']} x{i['num_threads']}" for i in pools})
    env = {
        "runtime": f"Python {platform.python_version()}, NumPy {np.__version__},"
        f" SciPy {scipy.__version__}, BLAS {', '.join(blas) or 'unknown'}",
        "module": str(Path(ub.__file__).parent),
    }
    print(json.dumps({"env": env, "results": results}))


def _run(fn, args, variants, data, name, repeat) -> dict:
    """Time each variant (one warm-up call first); keep the outputs of the last run."""
    fn(*args, **_resolved(variants[0], data, name))
    seconds, outputs = [], []
    for options in variants:
        options = _resolved(options, data, name)
        for _ in range(repeat if len(variants) == 1 else 1):
            start = time.perf_counter()
            result = fn(*args, **options)
            seconds.append(time.perf_counter() - start)
        outputs.append(_summary(result))
    return {"seconds": seconds, "outputs": outputs}


def _resolved(options, data, name):
    return {k: data.resolve(name, v) for k, v in options.items()}


def _summary(result) -> dict:
    if isinstance(result, dict):
        return result
    if hasattr(result, "centers"):
        return {
            "labels": result.labels.tolist(),
            "centers": result.centers.tolist(),
            "n_iter": int(result.n_iter),
        }
    if hasattr(result, "best_value"):
        return {"value": float(result.best_value)}
    return {"value": float(result)}


def _ubukit(ub) -> dict:
    def tpe_sphere(n_trials, *, seed):
        space = {"x": ub.uniform(-5, 5), "y": ub.uniform(-5, 5)}
        return ub.minimize(lambda p: (p["x"] - 1) ** 2 + p["y"] ** 2, space, n_trials, seed=seed)

    # Every public function, so a commit compared with --compare may lack some.
    functions = {name: getattr(ub, name) for name in ub.__all__ if name.islower()}
    return functions | {"tpe_sphere": tpe_sphere}


def _references() -> dict:
    """scikit-learn computations of the same quantities, called like UbuKit."""
    from sklearn.cluster import KMeans
    from sklearn.manifold import trustworthiness
    from sklearn.metrics import adjusted_mutual_info_score

    def kmeans(X, k, *, init=None, seed=None):
        model = KMeans(
            k,
            init="k-means++" if init is None else init,
            n_init=1,
            algorithm="lloyd",
            tol=0.0,
            max_iter=300,
            random_state=seed,
        ).fit(X)
        return {
            "labels": model.labels_.tolist(),
            "centers": model.cluster_centers_.tolist(),
            "n_iter": int(model.n_iter_),
        }

    return {
        "kmeans": kmeans,
        "trustworthiness": lambda X, Y, k: trustworthiness(X, Y, n_neighbors=k),
        "ami": adjusted_mutual_info_score,
    }


class _Data:
    """Datasets written by bench/run.py: <dir>/<dataset>/<field>.bin + manifest.json."""

    def __init__(self, root: Path, np):
        self.root, self.np = root, np
        self.manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self.cache = {}

    def resolve(self, name, value):
        fields = self.manifest.get(name, {})
        if not isinstance(value, str) or value not in fields:
            return tuple(value) if isinstance(value, list) else value
        if (name, value) not in self.cache:
            spec = fields[value]
            array = self.np.fromfile(self.root / name / f"{value}.bin", dtype=spec["dtype"])
            self.cache[name, value] = array.reshape(spec["shape"])
        return self.cache[name, value]


if __name__ == "__main__":
    main(sys.argv[1])
