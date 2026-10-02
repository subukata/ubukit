# License and release review

Project-owned UbuKit contributions use the unmodified [MIT License](../LICENSE),
with copyright (c) 2026 Seiki Ubukata. [License scope](../LICENSE-SCOPE.txt)
preserves the original SOM/scikit-learn terms and separate dataset notices.

## Packaging declarations

- Python and JavaScript both declare `MIT AND BSD-3-Clause`, reflecting project
  MIT material and the scoped BSD-3-Clause compatibility expressions. Each
  package carries its six license/notice files. The declarations use standard
  [npm SPDX expressions](https://docs.npmjs.com/cli/v11/configuring-npm/package-json/#license)
  and the [Python license field](https://packaging.python.org/en/latest/specifications/pyproject-toml/#license)
- Project-developed WASM kernels are covered by the project MIT selection.
  The earlier metric and historical RMCM exclusions were based on incomplete
  source capture; the maintainer's 2026-10-02 development clarification and the
  preserved development trail are recorded in
  [WASM development and source recovery](../javascript/wasm/DEVELOPMENT.md)
- UCI-derived label fixtures have their own
  [data attribution](../javascript/fixtures/DATASET_ATTRIBUTION.md). They are not
  npm runtime inputs. Their CC BY 4.0 notices are not a software license

## Source preservation and release checks

Metric WAT is reconstructed from its retained binary; the older RMCM source is
reconstructed by reversing documented later edits to preserved project WAT.
These labels remain explicit even when compilation reproduces every byte.
The maintainer's project-development clarification and the byte-rebuild checks
answer different questions; neither is presented as a legal determination.
There is no remaining external-origin exclusion based solely on these missing
historical source files. Preserve the private elite implementations and all
original build evidence.

The named Mulberry32, Acklam and A&S references must continue to distinguish
algorithm/formula references from any directly borrowed implementation and its
applicable notice. Existing external licenses are not replaced by a project
source declaration. Dependencies retain their own terms.

Before public distribution, verify the final artifacts' notices and source
records and obtain publication approval. License selection and alpha package
metadata do not authorize a registry upload or repository visibility change.
