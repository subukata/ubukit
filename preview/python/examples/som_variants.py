"""Run with the installed private dev4 wheel; no plotting dependency needed."""
import numpy as np
from ubukit import som, som_batch, initialize_som, initialize_som_batch

rng = np.random.default_rng(9)
X = np.concatenate((rng.normal(-1, .25, (50, 4)), rng.normal(1, .25, (50, 4))))
# Seeded row sampling makes both variants start from the same prototypes.
options = dict(grid_shape=(16, 16), random_state=42, epochs=5)
for method in (som, som_batch):
    result = method(X, **options)
    print(result['algorithm'], result['centers'].shape,
          result['iterations'], result['unit'], result['epochs_completed'])
    assert result['centers'].shape == (256, 4)
    assert result['embedding'].shape == (100, 2)

# A UI advances one committed learning update per displayed frame.
for initialize in (initialize_som, initialize_som_batch):
    state = initialize(X, **options)
    frame = state.step()
    print('frame:', frame['algorithm'], frame['iterations'], frame['unit'])
    # Online: one of N samples, epoch counter still 0.
    # Batch: all N samples assigned to the old prototypes, then epoch 1 committed.
    assert frame['iterations'] == 1
    assert frame['epochs_completed'] == (0 if frame['unit'] == 'sample' else 1)
    frame['centers'][:] = 99  # Snapshot mutation never edits state.
    assert not np.all(state.centers == 99)
    state.cancel()
    assert state.step() is None
