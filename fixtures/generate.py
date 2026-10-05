"""Generate fixtures.json: Python reference results that the JavaScript port must reproduce.

Every case fixes its initialization explicitly (or uses deterministic PCA), so
both languages walk the same iterates. Run from the repository root:

    python fixtures/generate.py
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import numpy as np

import ubukit as ub

PATH = Path(__file__).with_name("fixtures.json")


def _datasets() -> dict:
    rng = np.random.default_rng(1)
    blobs = np.vstack([rng.normal(c, 0.6, (20, 2)) for c in ([0, 0], [4, 0], [1, 4])])
    grid_init = np.column_stack(
        [np.tile(np.linspace(-1, 5, 4), 3), np.repeat(np.linspace(-1, 5, 3), 4)]
    )
    labels = np.random.default_rng(2)
    correlated = np.random.default_rng(6).normal(size=(60, 2)) @ [[1.0, 0.0], [0.6, 0.8]]
    return {
        "blobs": blobs,
        "centers": np.array([[0.5, 0.5], [3.0, 1.0], [1.0, 3.0]]),
        "grid_init": grid_init,
        "high": np.random.default_rng(3).normal(size=(40, 5)),
        "low": np.random.default_rng(4).normal(size=(40, 2)),
        "wide": np.random.default_rng(5).normal(size=(30, 100)) * np.linspace(3, 0.1, 100),
        "standard": (correlated - correlated.mean(axis=0)) / correlated.std(axis=0),
        "labels_a": labels.integers(0, 4, 50),
        "labels_b": labels.integers(0, 3, 50),
        # Clusters large enough that the expected-MI sums are windowed.
        "labels_c": labels.integers(0, 2, 400),
        "labels_d": labels.integers(0, 3, 400),
    }


# (method, data, positional args, options); strings in args/options name datasets.
CASES = [
    ("kmeans", "blobs", [3], {"init": "centers"}),
    ("fcm", "blobs", [3], {"init": "centers", "m": 2.0, "tol": 1e-10}),
    ("fcm", "blobs", [3], {"init": "centers", "m": 1.5, "tol": 1e-10}),
    ("efcm", "blobs", [3], {"init": "centers", "tau": 2.0, "tol": 1e-10}),
    ("rcm", "blobs", [3], {"init": "centers", "alpha": 1.3}),
    ("rcm", "blobs", [3], {"init": "centers", "alpha": 1.2, "beta": 0.4, "p": 2.0}),
    ("rmcm", "blobs", [3, 0.8], {"init": "centers"}),
    ("batch_som", "blobs", [[3, 4]], {"init": "grid_init", "epochs": 8}),
    ("batch_som", "blobs", [[3, 4]], {"epochs": 8}),
    ("som", "blobs", [[3, 4]], {"init": "grid_init", "epochs": 2, "shuffle": False}),
    ("som_olp", "blobs", [[3, 4]], {"lam": 0.5, "gamma": 1.0, "tol": 1e-10}),
    # PCA initialization with D > 64 > N, and with axes whose components tie.
    ("som_olp", "wide", [[3, 4]], {"lam": 50.0, "gamma": 1.0, "max_iter": 5}),
    ("som_olp", "standard", [[3, 4]], {"lam": 0.5, "gamma": 1.0, "max_iter": 5}),
]
METRICS = [
    ("ari", ["labels_a", "labels_b"], {}),
    *[
        ("ami", ["labels_a", "labels_b"], {"average": a})
        for a in ("arithmetic", "geometric", "min", "max")
    ],
    ("ami", ["labels_c", "labels_d"], {}),
    ("trustworthiness", ["high", "low", 5], {}),
    ("continuity", ["high", "low", 5], {}),
]


def _resolve(value, data):
    return data[value] if isinstance(value, str) and value in data else value


def build() -> dict:
    data = _datasets()
    cases = []
    for method, X, args, options in CASES:
        r = getattr(ub, method)(
            data[X],
            *[tuple(a) if isinstance(a, list) else a for a in args],
            **{k: _resolve(v, data) for k, v in options.items()},
        )
        expect = {"centers": r.centers, "labels": r.labels}
        if r.membership is not None:
            expect["membership"] = r.membership
        if r.embedding is not None:
            expect["embedding"] = r.embedding
        cases.append({"method": method, "X": X, "args": args, "options": options, "expect": expect})
    for method, args, options in METRICS:
        value = getattr(ub, method)(*[_resolve(a, data) for a in args], **options)
        cases.append({"method": method, "args": args, "options": options, "expect": value})
    return json.loads(json.dumps({"data": data, "cases": cases}, default=_jsonable))


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value))


def dumps(fixtures: dict) -> str:
    """JSON with one dataset or case per line, so a diff names the cases that changed."""
    compact = partial(json.dumps, separators=(",", ":"))
    data = ",\n".join(f"{json.dumps(k)}:{compact(v)}" for k, v in fixtures["data"].items())
    cases = ",\n".join(compact(case) for case in fixtures["cases"])
    return f'{{"data":{{\n{data}\n}},\n"cases":[\n{cases}\n]}}\n'


if __name__ == "__main__":
    PATH.write_text(dumps(build()), encoding="utf-8", newline="\n")
    print(f"wrote {PATH}")
