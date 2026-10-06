"""Benchmark cases and their data. To benchmark a new algorithm, add a case.

A case calls a UbuKit function (snake_case; JavaScript uses camelCase) with
arguments and options in which strings naming a field of the case's dataset
are replaced by that field. Datasets are generated from a fixed seed at the
base size times the size factor (--size), so every run sees the same data.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Case:
    name: str
    method: str
    data: str
    args: tuple
    options: dict = field(default_factory=dict)
    # Measures computed from the result: "ari" (against the dataset's y),
    # "qe" (mean squared distance to the nearest prototype) or "value".
    quality: tuple = ("qe",)
    # Deterministic work: Python and JavaScript must give the same quality.
    parity: bool = True
    # With --quality, run the default initialization over several seeds
    # (drop "init", vary "seed") instead of the fixed work.
    seeded: bool = False


CLUSTERING = ("ari", "qe")
GIVEN = {"init": "init"}  # the dataset's fixed initial centers

CASES = [
    Case("kmeans", "kmeans", "blobs", ("X", 10), GIVEN, CLUSTERING, seeded=True),
    Case("kmeans overlap", "kmeans", "overlap", ("X", 10), GIVEN, CLUSTERING, seeded=True),
    Case("kmeans noise", "kmeans", "noise", ("X", 10), GIVEN, ("qe",), seeded=True),
    Case("kmeans K=100", "kmeans", "blobs100", ("X", 100), GIVEN, CLUSTERING, seeded=True),
    Case("fcm", "fcm", "blobs", ("X", 10), GIVEN, CLUSTERING, seeded=True),
    Case("fcm overlap", "fcm", "overlap", ("X", 10), GIVEN, CLUSTERING, seeded=True),
    Case("efcm", "efcm", "blobs", ("X", 10), GIVEN | {"tau": 5.0}, CLUSTERING, seeded=True),
    Case("rcm", "rcm", "blobs", ("X", 10), GIVEN | {"alpha": 1.2}, CLUSTERING, seeded=True),
    Case("rmcm", "rmcm", "small", ("X", 10, 3.0), GIVEN, CLUSTERING, seeded=True),
    Case("batch_som", "batch_som", "blobs", ("X", (10, 10)), {"epochs": 20}),
    Case(
        "som", "som", "small", ("X", (10, 10)), {"epochs": 2, "seed": 0}, parity=False, seeded=True
    ),
    # A large map: memory or time that grows with the square of the units shows here.
    Case("som 40x40", "som", "small", ("X", (40, 40)), {"epochs": 1, "seed": 0}, parity=False),
    Case(
        "som_olp",
        "som_olp",
        "small",
        ("X", (10, 10)),
        {"lam": 5.0, "gamma": 1.0, "max_iter": 30, "tol": 0.0},
    ),
    Case("trustworthiness", "trustworthiness", "embed", ("X", "Y", 10), quality=("value",)),
    Case("ari", "ari", "labels", ("a", "b"), quality=("value",)),
    Case("ami", "ami", "labels", ("a", "b"), quality=("value",)),
    Case("tpe", "tpe_sphere", "none", (200,), {"seed": 0}, ("value",), parity=False),
    # Python's compiled kernels (engine="numba"); only Python runs these.
    Case(
        "som numba",
        "som",
        "small",
        ("X", (10, 10)),
        {"epochs": 2, "seed": 0, "engine": "numba"},
        parity=False,
        seeded=True,
    ),
    Case(
        "trustworthiness numba",
        "trustworthiness",
        "embed",
        ("X", "Y", 10),
        {"engine": "numba"},
        ("value",),
    ),
    Case("ami numba", "ami", "labels", ("a", "b"), {"engine": "numba"}, ("value",)),
]


def _blobs(n, rng, k=10, spread=5.0):
    centers = rng.uniform(-spread, spread, (k, 16))
    y = rng.permutation(np.arange(n) % k)
    X = centers[y] + rng.normal(size=(n, 16))
    return {"X": X, "y": y, "init": X[rng.choice(n, k, replace=False)] + 0.01}


def _noise(n, rng):
    X = rng.uniform(-1, 1, (n, 16))
    return {"X": X, "init": X[rng.choice(n, 10, replace=False)] + 0.01}


def _embed(n, rng):
    X = rng.normal(size=(n, 16))
    return {"X": X, "Y": X[:, :2] + 0.3 * rng.normal(size=(n, 2))}


def _labels(n, rng):
    return {"a": rng.integers(0, 50, n), "b": rng.integers(0, 40, n)}


# name: (size at factor 1, generator)
DATA = {
    "blobs": (20_000, _blobs),
    "overlap": (20_000, lambda n, rng: _blobs(n, rng, spread=1.0)),
    "noise": (20_000, _noise),
    "blobs100": (20_000, lambda n, rng: _blobs(n, rng, k=100)),
    "small": (5_000, _blobs),
    "embed": (3_000, _embed),
    "labels": (1_000_000, _labels),
    "none": (0, lambda n, rng: {}),
}
SIZES = {"smoke": 0.03, "s": 0.25, "m": 1.0, "l": 4.0}


def dataset(name: str, factor: float) -> tuple[int, dict]:
    """Return (n, fields) of a dataset; the seed depends only on its name."""
    base, generate = DATA[name]
    n = max(int(base * factor), 120) if base else 0
    return n, generate(n, np.random.default_rng(zlib.crc32(name.encode())))
