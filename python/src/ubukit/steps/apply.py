"""The functions of ``ubukit.apply`` as generators: trained prototypes, fixed,
applied to new data.

SOM-OLP's iterations are observable here and can take new rows by ``send``
(the same points, moved); the other methods assign once.
"""

from .._cluster import apply_efcm as efcm
from .._cluster import apply_fcm as fcm
from .._cluster import apply_kmeans as kmeans
from .._cluster import apply_rcm as rcm
from .._cluster import apply_rmcm as rmcm
from .._som import apply_batch_som as batch_som
from .._som import apply_som as som
from .._som import apply_som_olp as som_olp

__all__ = ["batch_som", "efcm", "fcm", "kmeans", "rcm", "rmcm", "som", "som_olp"]
