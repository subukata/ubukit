# Dataset-derived label fixtures

This notice applies to the cases named `load_iris`, `load_wine`, and
`load_breast_cancer` in both `sklearn-1.8.json` and `benchmark-inputs.json`.
It does not describe the synthetic cases in those files as UCI datasets.

## Included content and generation

Each named case contains the dataset's target-label vector (`a`) and a derived
clustering-label vector (`b`). The reference panel also records adjusted Rand
and adjusted mutual information scores. These JSON files do **not** contain the
original numeric feature matrices, source record identifiers, or raw images.
The label pairs in the two files are identical.

`../tests/generate_references.py` records the source as scikit-learn 1.8.0 loaders.
It preserves loader target order/codes and generates predictions using
`StandardScaler` followed by `KMeans` with the number of clusters set to the
number of target classes, `n_init=10`, and `random_state=0`. Results were exported
to JSON. The fixture metadata records NumPy 2.3.5. Benchmark cases retain the
same label pairs without the reference scores.

## Sources and attribution

The following official UCI pages identify these datasets as
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
Source and license metadata were checked on 2026-10-02.

- **Iris:** R. A. Fisher (1936), *Iris*, UCI Machine Learning Repository,
  [DOI: 10.24432/C56C76](https://doi.org/10.24432/C56C76).
  [UCI dataset 53](https://archive.ics.uci.edu/dataset/53/iris);
  [scikit-learn loader](https://scikit-learn.org/1.8/modules/generated/sklearn.datasets.load_iris.html).
  The fixture contains 150 target labels and 150 derived predictions. The
  generator uses sklearn's Fisher-corrected feature variant for clustering;
  the JSON fixture contains labels rather than those feature values.
- **Wine:** Stefan Aeberhard and M. Forina (1992), *Wine*, UCI Machine Learning
  Repository, [DOI: 10.24432/C5PC7J](https://doi.org/10.24432/C5PC7J).
  [UCI dataset 109](https://archive.ics.uci.edu/dataset/109/wine);
  [scikit-learn loader](https://scikit-learn.org/1.8/modules/generated/sklearn.datasets.load_wine.html).
  The fixture contains 178 target labels and 178 derived predictions. Sklearn
  target codes 0/1/2 correspond to UCI class codes 1/2/3.
- **Breast Cancer Wisconsin (Diagnostic):** William Wolberg, Olvi Mangasarian,
  Nick Street, and W. Street (1993), UCI Machine Learning Repository,
  [DOI: 10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B).
  [UCI dataset 17](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic);
  [scikit-learn loader](https://scikit-learn.org/1.8/modules/generated/sklearn.datasets.load_breast_cancer.html).
  The fixture contains 569 target labels and 569 derived predictions. Sklearn
  target code 0 denotes malignant and 1 benign. These are public benchmark
  labels; no private patient records were supplied. This is an algorithm test
  fixture, not a clinical diagnostic tool.

Keep this source, attribution, license-link, and modification information with
copies of these dataset-derived fixtures. The separate scikit-learn software
notice is in `../LICENSE-SCIKIT-LEARN.txt`; it is not the dataset license.
This data notice does not select a license for UbuKit's project code.
