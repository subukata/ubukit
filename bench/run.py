"""Time and score UbuKit in Python and JavaScript on the cases in bench/cases.py.

Run from the repository root, with Node.js on PATH and the Python dev group
installed (scikit-learn provides the reference column):

    python bench/run.py                   # every case at size m
    python bench/run.py --only kmeans     # cases whose name contains "kmeans"
    python bench/run.py --compare main    # this tree against a git ref
    python bench/run.py --size s,m,l      # growth with N
    python bench/run.py --quality         # default initialization over seeds
    python bench/run.py --baseline        # also the textbook code, bench/baseline.py
    python bench/run.py --size smoke --check

Each implementation runs in its own process (bench/worker.py, worker.mjs) on
the same data, written once as binary files. A time is the median of
--repeat runs after a warm-up, shown with +- half the range. --compare runs
the two trees in ABBA order and calls a difference only when their samples
do not overlap and their medians differ by at least 5%. --baseline adds the
time of the textbook code and each time's speed-up over it, and marks a
result that differs from it. Quality is computed here, the same way for
every implementation.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import median

import numpy as np
from cases import CASES, SIZES, dataset
from scipy.spatial.distance import cdist

import ubukit as ub

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "bench"
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
MIN_EFFECT = 0.05


def main() -> None:
    opts = _parse()
    cases = [c for c in CASES if not opts.only or any(w in c.name for w in opts.only.split(","))]
    seeds = list(range(opts.seeds)) if opts.quality else None
    machine = f"{platform.system()} {platform.machine()}, {os.cpu_count()} CPUs"
    report = {"machine": machine, "rows": []}
    with tempfile.TemporaryDirectory(prefix="ubukit-bench-") as tmp:
        runner = _Runner(Path(tmp), opts.threads)
        with _tree(opts.compare, Path(tmp)) as base:
            for size in opts.size.split(","):
                fields = _write_data(runner.data_dir(size), {c.data for c in cases}, SIZES[size])
                if base:
                    report["rows"] += _compare(runner, size, cases, fields, base, opts)
                else:
                    report["rows"] += _measure(runner, size, cases, fields, seeds, opts)
        report["env"] = runner.env
    _print(report, opts)
    if opts.json:
        Path(opts.json).write_text(json.dumps(report, indent=1), encoding="utf-8")
    failures = [r for r in report["rows"] if r.get("parity") is False]
    if opts.check and failures:
        sys.exit("Python and JavaScript disagree on: " + ", ".join(r["case"] for r in failures))


def _measure(runner, size, cases, fields, seeds, opts) -> list[dict]:
    impls = {
        "python": runner.work("python", ROOT, size, cases, opts.repeat, seeds, memory=True),
        "javascript": runner.work("javascript", ROOT, size, cases, opts.repeat, seeds),
        "scikit-learn": runner.work("scikit-learn", ROOT, size, cases, opts.repeat, seeds),
    }
    if opts.baseline:
        impls["baseline"] = runner.work("baseline", ROOT, size, cases, min(opts.repeat, 3), None)
    rows = []
    for case in cases:
        row = {"case": case.name, "size": size, "n": fields[case.data][0]}
        for impl, results in impls.items():
            if case.name in results:
                row[impl] = _scored(case, results[case.name], fields)
        py, js, base = (row.get(i, {}).get("quality") for i in ("python", "javascript", "baseline"))
        if case.parity and not seeds and py and js:
            row["parity"] = _same(py, js, 1e-6)
        if py and base:
            row["baseline agrees"] = _same(py, base, 1e-6)
        rows.append(row)
    return rows


def _compare(runner, size, cases, fields, base, opts) -> list[dict]:
    samples = {}
    for r in range(opts.rounds):
        # ABBA order cancels slow drifts of the machine.
        for tree in (("base", base), ("new", ROOT))[:: 1 if r % 2 == 0 else -1]:
            for impl in ("python", "javascript"):
                results = runner.work(impl, tree[1], size, cases, opts.repeat, None)
                for name, result in results.items():
                    entry = samples.setdefault((name, impl, tree[0]), {"seconds": []})
                    entry["seconds"] += result.get("seconds", [])
                    entry["last"] = result
    rows = []
    for case in cases:
        for impl in ("python", "javascript"):
            old, new = samples.get((case.name, impl, "base")), samples.get((case.name, impl, "new"))
            if not (old and new):
                continue
            a = _scored(case, {**old["last"], "seconds": old["seconds"]}, fields)
            b = _scored(case, {**new["last"], "seconds": new["seconds"]}, fields)
            row = {"case": case.name, "size": size, "n": fields[case.data][0], "impl": impl}
            row |= {"base": a, "new": b}
            if "error" not in a and "error" not in b:
                ta, tb = old["seconds"], new["seconds"]
                ratio = median(tb) / median(ta)
                # A difference needs separated samples and at least MIN_EFFECT:
                # identical trees still differ by a few percent between runs.
                verdict = "same"
                if (max(tb) < min(ta) or min(tb) > max(ta)) and abs(ratio - 1) >= MIN_EFFECT:
                    verdict = "faster" if ratio < 1 else "slower"
                row |= {"ratio": ratio, "verdict": verdict}
                row["results"] = "same" if _same(a["quality"], b["quality"], 1e-9) else "changed"
            rows.append(row)
    return rows


def _scored(case, result, fields) -> dict:
    """Median time, spread, iterations and quality of one implementation's result."""
    if "error" in result:
        return result
    seconds, data = result["seconds"], fields[case.data][1]
    measures = [_quality(case, out, data) for out in result["outputs"]]
    keys = list(measures[0])
    out = {
        "seconds": median(seconds),
        # Seeded runs do different work, so only repeated runs have a spread.
        "spread": None
        if len(result["outputs"]) > 1 or len(seconds) < 2
        else (max(seconds) - min(seconds)) / median(seconds),
        "samples": seconds,
        "quality": {k: median(m[k] for m in measures) for k in keys},
        "range": {k: (min(m[k] for m in measures), max(m[k] for m in measures)) for k in keys},
    }
    iters = [o["n_iter"] for o in result["outputs"] if "n_iter" in o]
    if iters:
        out["n_iter"] = median(iters)
    if "peak_mb" in result:
        out["peak_mb"] = result["peak_mb"]
    return out


def _quality(case, output, data) -> dict:
    measures = {}
    for kind in case.quality:
        if kind == "ari":
            measures["ARI"] = ub.ari(data["y"], np.asarray(output["labels"]))
        elif kind == "qe":
            centers = np.asarray(output["centers"])
            measures["QE"] = float(cdist(data["X"], centers, "sqeuclidean").min(axis=1).mean())
        else:
            measures["value"] = output["value"]
    return measures


def _same(a: dict, b: dict, rtol: float) -> bool:
    return all(math.isclose(a[k], b[k], rel_tol=rtol, abs_tol=1e-12) for k in a)


class _Runner:
    """Launches workers on the data of each size and collects their environments."""

    def __init__(self, tmp: Path, threads: int | None):
        self.tmp, self.env, self.calls = tmp, {}, 0
        limits = dict.fromkeys(THREAD_VARIABLES, str(threads)) if threads else {}
        self.environ = os.environ | limits

    def data_dir(self, size: str) -> Path:
        return self.tmp / f"data-{size}"

    def work(self, impl, src, size, cases, repeat, seeds, memory=False) -> dict:
        if impl not in ("python", "baseline"):  # engines are Python's; the baseline ignores them
            cases = [c for c in cases if "engine" not in c.options]
        self.calls += 1
        request = self.tmp / f"request-{self.calls}.json"
        payload = {"impl": impl, "src": str(src), "data": str(self.data_dir(size))}
        payload |= {"repeat": repeat, "seeds": seeds, "memory": memory}
        payload["cases"] = [
            {"name": c.name, "method": c.method, "data": c.data, "args": c.args}
            | {"options": c.options, "seeded": c.seeded}
            for c in cases
        ]
        request.write_text(json.dumps(payload), encoding="utf-8")
        worker = ["node", str(HERE / "worker.mjs")] if impl == "javascript" else None
        command = (worker or [sys.executable, str(HERE / "worker.py")]) + [str(request)]
        done = subprocess.run(command, capture_output=True, text=True, env=self.environ)
        if done.returncode:
            sys.exit(f"{impl} worker failed:\n{done.stderr}")
        output = json.loads(done.stdout)
        self.env[impl] = output["env"]["runtime"]
        return output["results"]


@contextlib.contextmanager
def _tree(ref: str | None, tmp: Path):
    """A temporary git worktree of ref (None: no comparison)."""
    if not ref:
        yield None
        return
    path = tmp / "base"
    git = ["git", "-C", str(ROOT), "worktree"]
    subprocess.run([*git, "add", "--detach", "--quiet", str(path), ref], check=True)
    try:
        yield path
    finally:
        subprocess.run([*git, "remove", "--force", str(path)], check=True)


def _write_data(root: Path, names: set, factor: float) -> dict:
    """Write each dataset's fields as raw binary; return {name: (n, fields)}."""
    manifest, fields = {}, {}
    for name in sorted(names):
        n, data = dataset(name, factor)
        fields[name] = (n, data)
        (root / name).mkdir(parents=True, exist_ok=True)
        manifest[name] = {}
        for key, value in data.items():
            dtype = np.int32 if value.dtype.kind == "i" else np.float64
            array = np.ascontiguousarray(value, dtype=dtype)
            array.tofile(root / name / f"{key}.bin")
            manifest[name][key] = {"dtype": array.dtype.name, "shape": list(array.shape)}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return fields


def _print(report: dict, opts) -> None:
    env = report["env"]
    print("; ".join([report["machine"], *(env[k] for k in ("python", "javascript") if k in env)]))
    print(f"--repeat {opts.repeat}" + (f", --threads {opts.threads}" if opts.threads else ""))
    print()
    if opts.compare:
        print(f"| case | N | | {opts.compare} s | new s | new/base | verdict | results |")
        print("|---|---:|---|---:|---:|---:|---|---|")
        for r in report["rows"]:
            a, b = r["base"], r["new"]
            if "ratio" not in r:
                print(f"| {r['case']} | {r['n']} | {r['impl']} | {_error(a)} | {_error(b)} | | | |")
                continue
            print(
                f"| {r['case']} | {r['n']} | {r['impl']} | {_time(a)} | {_time(b)}"
                f" | {r['ratio']:.2f} | {r['verdict']} | {_results(r)} |"
            )
        return
    first = ["baseline s"] if opts.baseline else []
    order = "base/Py/JS/sk" if opts.baseline else "Py/JS/sk"
    print(
        "| "
        + " | ".join(["case", "N", *first, "Python s", "JS s", "scikit-learn s"])
        + f" | iterations ({order}) | Python MB | Python quality | JS quality |"
    )
    print("|---|---:|" + "---:|" * len(first) + "---:|---:|---:|---:|---:|---|---|")
    for r in report["rows"]:
        py, js, sk = r.get("python", {}), r.get("javascript", {}), r.get("scikit-learn")
        base = r.get("baseline")
        runs = ([base or {}] if opts.baseline else []) + [py, js, sk or {}]
        iters = "/".join(f"{d['n_iter']:g}" for d in runs if "n_iter" in d)
        mb = f"{py['peak_mb']:.0f}" if "peak_mb" in py else ""
        mark = " **(differs)**" if r.get("parity") is False else ""
        if r.get("baseline agrees") is False:
            mark += " **(baseline differs)**"
        # Each time with its speed-up over the baseline; only Python has the Numba cases.
        times = [_time(d) + _speedup(base, d) if d else "" for d in (py, js, sk)]
        cells = [r["case"], r["n"] or "", *([_time(base) if base else ""] if opts.baseline else [])]
        cells += [*times, iters, mb, _quality_text(py), _quality_text(js) + mark]
        print("| " + " | ".join(map(str, cells)) + " |")
    _print_growth(report["rows"])


def _speedup(base: dict | None, d: dict) -> str:
    if not base or "seconds" not in base or "seconds" not in d:
        return ""
    return f" ({base['seconds'] / d['seconds']:.1f}x)"


def _print_growth(rows: list[dict]) -> None:
    """With several sizes: exponent b of time ~ N^b between the smallest and largest N,
    per iteration for iterative methods (their iteration counts change with N)."""
    by_case = {}
    for r in rows:
        by_case.setdefault(r["case"], []).append(r)
    lines = []
    for name, rs in by_case.items():
        lo, hi = min(rs, key=lambda r: r["n"]), max(rs, key=lambda r: r["n"])
        if hi["n"] <= lo["n"]:
            continue
        exps = []
        for impl in ("python", "javascript"):
            a, b = lo.get(impl, {}), hi.get(impl, {})
            t0, t1 = a.get("seconds"), b.get("seconds")
            if t0 and t1:
                t0, t1 = t0 / a.get("n_iter", 1), t1 / b.get("n_iter", 1)
                exps.append(f"{impl} {math.log(t1 / t0) / math.log(hi['n'] / lo['n']):.2f}")
        lines.append(f"| {name} | {lo['n']} -> {hi['n']} | {', '.join(exps)} |")
    if lines:
        print("\n| case | N | growth exponent (time ~ N^b) |\n|---|---|---|")
        print("\n".join(lines))


def _time(d: dict) -> str:
    if "error" in d:
        return _error(d)
    spread = f" +-{d['spread'] / 2:.0%}" if d.get("spread") is not None else ""
    return f"{d['seconds']:.3f}{spread}"


def _quality_text(d: dict) -> str:
    if "quality" not in d:
        return _error(d) if d else ""
    parts = []
    for k, v in d["quality"].items():
        lo, hi = d["range"][k]
        parts.append(f"{k} {v:.4g}" + (f" [{lo:.4g}, {hi:.4g}]" if hi > lo else ""))
    return " ".join(parts)


def _results(r: dict) -> str:
    if r["results"] == "same":
        return "same"
    a, b = r["base"]["quality"], r["new"]["quality"]
    return "changed: " + ", ".join(f"{k} {a[k]:.4g} -> {b[k]:.4g}" for k in a)


def _error(d: dict) -> str:
    return "error" if "error" in d else ""


def _parse():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", help="comma-separated parts of case names")
    parser.add_argument("--size", default="m", help=f"comma-separated, from {', '.join(SIZES)}")
    parser.add_argument("--repeat", type=int, default=5, help="timed runs per case and process")
    parser.add_argument("--quality", action="store_true", help="default initialization over seeds")
    parser.add_argument("--seeds", type=int, default=5, help="seeds for --quality")
    parser.add_argument("--compare", metavar="REF", help="compare this tree with a git ref")
    parser.add_argument("--rounds", type=int, default=4, help="ABBA rounds for --compare (even)")
    parser.add_argument("--threads", type=int, help="BLAS threads for the Python processes")
    parser.add_argument("--check", action="store_true", help="fail if the languages disagree")
    parser.add_argument("--json", metavar="PATH", help="also write all samples as JSON")
    parser.add_argument("--baseline", action="store_true", help="also time bench/baseline.py")
    opts = parser.parse_args()
    if opts.baseline and (opts.quality or opts.compare or "l" in opts.size.split(",")):
        # The textbook code holds dense (N, N) arrays and supports only the given starts.
        parser.error("--baseline times the fixed work at sizes up to m")
    if opts.size == "smoke":
        opts.repeat, opts.rounds = 1, 1
    return opts


if __name__ == "__main__":
    main()
