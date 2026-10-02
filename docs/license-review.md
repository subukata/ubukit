# License and release review

Project-owned UbuKit contributions use the unmodified [MIT License](../LICENSE),
with copyright (c) 2026 Seiki Ubukata. This selection is scoped by
[LICENSE-SCOPE.txt](../LICENSE-SCOPE.txt); it does not replace original
SOM/scikit-learn terms or declare pending components cleared.

## Packaging declarations

- Python carries MIT and scoped BSD-3-Clause material and declares
  `MIT AND BSD-3-Clause`. All six license/notice files are packaged.
- JavaScript uses `SEE LICENSE IN LICENSE-SCOPE.txt`: the current tarball still
  includes the pending-origin metric WASM wrapper/binary. Its private flag stays
  enabled. Do not simplify this field to MIT while the exclusion remains.
- UCI-derived label fixtures have their own
  [data attribution](../javascript/fixtures/DATASET_ATTRIBUTION.md). They are not
  npm runtime inputs. Their CC BY 4.0 source notices are not a software license.

## Remaining source-origin gates

The metric WASM binary `b7b68848…` still lacks its original `distance_rows8.wat`.
Older RMCM `63768dc7…` occurs only in repository validation/rebuild evidence,
not current runtime packages. Both remain outside the project MIT grant.
Four other current kernels have recovered WAT, but byte equality alone does not
identify an original author. Preserve all source/build evidence and private
elite implementations while resolving contributor/source records.

The named Mulberry32, Acklam and A&S numerical references are not blanket claims
about copied source: primary upstream permissions and formula references must
be distinguished from the actual implementation history. The preserved
SOURCE_MANIFEST files record technical lineage, not legal ownership.

Before public distribution, resolve or separately exclude pending components,
review known collaborator/institution/borrowed-source interests, verify the
exact final distribution's notices and scope, and obtain publication approval.
Do not delete historical evidence to simplify a license inventory. Private
package flags, source snapshots and numerical contracts remain unchanged except
for the intentional license/notice metadata updates recorded with this change.
