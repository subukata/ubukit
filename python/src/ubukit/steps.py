"""The fitting functions as generators.

Each yields a Progress after every iteration (every epoch for the online and
batch SOM), whose ``result()`` is the Result as if the run had stopped
there; takes new rows through ``send`` between iterations; and returns the
final Result. ``ubukit.fcm`` and the other fitting functions run these to
the end.

    run = steps.som_olp(X, (10, 10), lam=0.5, gamma=1.0)
    for p in run:
        draw(p.result())
"""

from .cluster import efcm, fcm, kmeans, rcm, rmcm
from .som import batch_som, som, som_olp

__all__ = ["batch_som", "efcm", "fcm", "kmeans", "rcm", "rmcm", "som", "som_olp"]
