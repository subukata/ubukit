from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from somolp import SOMOLP

M_SIDE = 16

# Best hyperparameters found by Optuna ((TW+CN)/2, k=5, m=16*16, StandardScaler, pca_scale=2.0)
DATASETS = [
    ("iris",           load_iris,           dict(gamma=51.94,  lam=1.32)),
    ("wine",           load_wine,           dict(gamma=961.4,  lam=21.14)),
    ("breast_cancer",  load_breast_cancer,  dict(gamma=807.2,  lam=13.49)),
    ("digits",         load_digits,         dict(gamma=5.53,   lam=2.79)),
]


def grid_segments(R, m):
    G = R.reshape(m, m, 2)
    h = np.stack([G[:, :-1], G[:, 1:]], axis=2).reshape(-1, 2, 2)
    v = np.stack([G[:-1], G[1:]], axis=2).reshape(-1, 2, 2)
    return np.concatenate([h, v], axis=0)


def run(name, loader, params):
    X, y = loader(return_X_y=True)
    X = StandardScaler().fit_transform(X)

    t = np.linspace(-1, 1, M_SIDE)
    gx, gy = np.meshgrid(t, t)
    R = np.column_stack([gx.ravel(), gy.ravel()])

    model = SOMOLP(
        R, gamma=params["gamma"], lam=params["lam"],
        max_iters=1000, tol=1e-4,
    ).fit(X)
    print(f"[{name} Grid {M_SIDE}x{M_SIDE}] n_iter={model.n_iter}  obj={model.history[-1]:.6f}")

    save_path = Path(__file__).resolve().parent / "results" / f"som-olp_{name}.png"
    segments = grid_segments(model.R, M_SIDE)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.add_collection(LineCollection(segments, colors="#cccccc", linewidths=0.4, zorder=1))
    ax.scatter(model.R[:, 0], model.R[:, 1], s=8, c="#aaaaaa", zorder=2)
    ax.scatter(model.V[:, 0], model.V[:, 1], s=12, c=y, cmap="tab10", vmin=0, vmax=9, zorder=3)
    ax.set_title(f"{name} — Latent space (Grid {M_SIDE}x{M_SIDE}, M={M_SIDE*M_SIDE})")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Saved: {save_path}")


def main():
    for name, loader, params in DATASETS:
        run(name, loader, params)


if __name__ == "__main__":
    main()
