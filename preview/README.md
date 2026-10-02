# UbuKit private preview: Python dev5 / JavaScript dev6

This is the current integrated source snapshot. Start with the repository
[README](../README.md), [REPRODUCE.md](REPRODUCE.md), and
[VERIFICATION.md](VERIFICATION.md).

- Python: `python/staging`, version `0.0.0.dev5`, 38 lazy facade exports
- JavaScript: `javascript/package`, version `0.1.0-dev.6`, seven export entrypoints
- Traditional online `som` and true frozen-BMU `som_batch`, with a default 16×16
  rectangular lattice and arbitrary feature dimension
- Stateful sample/epoch updates, initialization, schedules and owned results;
  [Python SOM API](python/staging/SOM.md), [JavaScript SOM API](javascript/package/SOM.md)
- Existing SOM-OLP, clustering, ARI/AMI and optional lightweight TPE/random APIs
- Installed-package regression harnesses with frozen test-only references

SOURCE_SNAPSHOT.json binds all 62 Python and 27 JavaScript runtime files.
VERIFICATION_SNAPSHOT.json binds tests, references and selected evidence.
See [SOM_ADDITION.md](SOM_ADDITION.md) for the added algorithms and evidence scope.
See [EFFICIENCY.md](EFFICIENCY.md) for the opt-in Python experiment, conservative JavaScript routing, original-route preservation, benchmark caveats and maintainability checks.
Existing ARI/AMI numerical sources and recorded measurements are unchanged.
Actual browser execution and other platform combinations remain unverified.

Package documentation was adapted for repository use, so rebuilt archive bytes
can differ from earlier standalone deliveries. No registry publication, new
project-wide license or broad performance guarantee is implied.

The old top-level source trees remain historical checkpoints. Use these preview
paths for new work. Read NUMERICAL_LIMITS.md before extreme inputs.
