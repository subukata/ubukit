from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,csv
base=Path(__file__).resolve().parent;p=base/'reports'
j=json.loads((p/'matched-node.json').read_text());py=json.loads((p/'matched-python.json').read_text())
assert j['fixtureSha256']==py['fixtureSha256']
all_results={**j['results'],**py['results']};ref=all_results['sklearn_1_8'];summary=[]
for implementation,values in all_results.items():
 for metric,data in values.items():
  if metric=='separate':continue
  value=data['value'];reference=ref[metric]['value']
  error=max(abs(value[k]-reference[k]) for k in value) if isinstance(value,dict) else abs(value-reference)
  summary.append(dict(implementation=implementation,metric=metric,n=10000,kTrue=50,kPred=50,medianMs=data['medianMs'],sklearnMedianMs=ref[metric]['medianMs'],speedupVsSklearn=ref[metric]['medianMs']/data['medianMs'],absoluteErrorVsSklearn=error,fixtureSha256=j['fixtureSha256'],normalization='arithmetic',sklearnJoint='two separate public calls',warmups=j['warmups'],samples=j['repeats'],callsPerSample=j['batchSize']))
compact=dict(measuredAtUTC=datetime.now(timezone.utc).isoformat(),fixture='fixtures/matched-10000-k50.json',fixtureSha256=j['fixtureSha256'],seed=j['seed'],versions={'js':j['packageVersion'],'node':j['node'],'pythonPackage':py['packageVersion'],'python':py['python'],'sklearn':py['sklearn'],'numpy':py['numpy'],'numba':py['numba']},threadSettings=py['threadSettings'],numbaFirstAMICallMs=py['numbaCold']['firstNumbaAMICallMs'],rows=summary)
(p/'matched-summary.json').write_text(json.dumps(compact,indent=2)+'\n')
with (p/'matched-summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
def fmt(x):return f'{x:.6f}'
rows=['| Implementation | ARI ms | AMI ms | Joint ms |','|---|---:|---:|---:|']
for name,d in all_results.items():rows.append('| '+name+' | '+' | '.join(fmt(d[m]['medianMs']) if m in d else 'same ARI as NumPy' for m in ['ari','ami','joint'])+' |')
audit=json.loads((base/'audit/audit-results.json').read_text());parity=json.loads((p/'sklearn-parity.json').read_text())
broadj=json.loads((p/'benchmark-node.json').read_text());broads={r['name']:r for r in json.loads((p/'benchmark-sklearn.json').read_text())['rows']}
broadrows=['| Case | JS joint ms | Ungrouped JS joint ms | sklearn two-call ms |','|---|---:|---:|---:|']
for r in broadj['rows']:broadrows.append(f"| {r['name']} | {r['times']['joint']['medianMs']:.6f} | {r['times']['baselineJoint']['medianMs']:.6f} | {broads[r['name']]['times']['joint']['medianMs']:.6f} |")
js=all_results['javascript_dev4'];np=all_results['python_dev3_numpy'];nb=all_results['python_dev3_numba_warm'];bl=all_results['javascript_ungrouped_baseline']
report=f'''# JavaScript ARI / AMI dev4 verification and measurements

## Result

Implemented `adjustedRandScore`, `adjustedMutualInfoScore`, and `adjustedScores`
at the package root and `ubukit-js/external-metrics`. Python-style name aliases
are included; options use camelCase. Runtime dependencies added: zero. The joint
API shares one encoding and contingency. The new package is private
`ubukit-js@0.1.0-dev.4`; verified JS dev3 and Python dev3 are preserved.

## Matched current-environment N=10,000 / K=50 panel

Same labels, seed {j['seed']}, SHA256:
`{j['fixtureSha256']}`

{chr(10).join(rows)}

Times are median milliseconds per invocation, from 20 warmups and 21 batches of
five calls. Input construction is excluded; public validation, label encoding
and contingency construction are included. Arithmetic AMI is used everywhere.
Sklearn's joint number is two separate public calls; the UbuKit joint APIs share
one contingency. No timing is reused from the earlier October 1 panel.

- JS joint is {ref['joint']['medianMs']/js['joint']['medianMs']:.2f}× faster than the sklearn two-call baseline on this case
- JS joint is {bl['joint']['medianMs']/js['joint']['medianMs']:.2f}× faster than the ungrouped JS comparison on this case
- JS remains {js['joint']['medianMs']/np['joint']['medianMs']:.2f}× slower than Python NumPy and {js['joint']['medianMs']/nb['joint']['medianMs']:.2f}× slower than warm Numba here
- JS joint vs its separate ARI+AMI calls: {js['separate']['medianMs']:.6f} → {js['joint']['medianMs']:.6f} ms ({js['separate']['medianMs']/js['joint']['medianMs']:.2f}×)
- Warm Numba excludes the separately measured first AMI call: {py['numbaCold']['firstNumbaAMICallMs']:.3f} ms, including compilation/cache loading
- ARI result: {js['joint']['value']['ari']:.16g}; AMI: {js['joint']['value']['ami']:.16g}
- JS versus sklearn differences: ARI zero; AMI {abs(js['joint']['value']['ami']-ref['joint']['value']['ami']):.3g}

Environment: {j['cpu']}, Linux {j['arch']}; Node {j['node']}; Python
{py['python']}, UbuKit Python {py['packageVersion']}, NumPy {py['numpy']},
scikit-learn {py['sklearn']}, Numba {py['numba']}. OMP, OpenBLAS, MKL and Numba
thread limits were all 1; Numba reported one thread. Python installed-distribution
RECORD ownership and its source hash were verified. Node used one synchronous
JS thread. The measurements ran serially after coordinating with other active
benchmarks. They are machine/workload-specific, not universal speed guarantees.

Raw per-sample times and score values: `reports/matched-node.json` and
`reports/matched-python.json`. Compact dashboard-ready rows:
`reports/matched-summary.json` / `.csv`. Exact labels:
`fixtures/matched-10000-k50.json`.

## Broader performance panel

{chr(10).join(broadrows)}

This separate exploratory panel uses three warmups and nine single-call samples,
so its tiny-input timings are noisier. All JS/Python rows use exactly the same
stored inputs and current environment. The straight JS baseline is a one-pass
Map contingency plus a scalar, full-support hypergeometric expectation per
cluster pair, without marginal-size compression. It is a meaningful numerical
baseline, not quadratic pair-label enumeration. It is faster than the candidate
on some skewed/small cases: grouping and BigInt/compensated bookkeeping have costs.
Repeated marginal sizes give the largest speedup. No blanket JS speedup is claimed.

## Verification

- Installed dev4 package: 890/890 regression tests pass, including inherited dev3 coverage
- Independent audit: {audit['counts']['ari']:,} ARI and {audit['counts']['ami']:,} AMI checks against exact-integer / 80-digit Decimal references
- {audit['counts']['shared']:,} joint checks and {audit['counts']['invariance']:,} swap/order/relabel invariance checks
- Independent AMI maximum absolute error: {audit['maximumAbsoluteErrors']['ami']:.3g}; ARI errors zero on those references
- 132 expected optional-min singular cases, 74 malformed/work-budget checks, early sample-limit and symmetric work-budget checks pass
- Large synthetic count tests check exact ARI products through N=4,294,967,292 and AMI internals through the 2**26 sample bound; these do not claim enormous public label arrays were allocated
- Extra sklearn 1.8 panel: {parity['cases']} cases including Iris/Wine/WDBC labels; largest AMI difference {parity['maxAMI']:.3g}, in a high-K case
- Pure ESM imports without dependency/host globals, isolated VM module execution and a real Node module worker pass
- Actual browser execution is not verified: cloud Chrome blocked the local test URL with `net::ERR_BLOCKED_BY_CLIENT`. The browser harness is included for independent execution

The independently reviewed runtime SHA256 is:
`{audit['runtimeSHA256']}`

## Numerical policy and edge cases

ARI keeps integer combinatorics exact with safe Number sums / BigInt products;
its final quotient is rounded to binary64. AMI uses a mode-centered, normalized,
full-support hypergeometric recurrence, grouped margin sizes and compensated
sums. Its equivalent conditional-entropy normalization avoids cancellation in
high-K partitions. There is no sampling or chosen tail cutoff, but binary64
underflow/rounding still exists and no universal absolute-error bound is claimed.

The default is arithmetic AMI. Empty, singleton and equivalent partitions return
1; constant-vs-nonconstant returns 0. Singleton-vs-other returns exact 0 for
arithmetic/geometric/max. Only optional min normalization in that case is truly
0/0 and throws a documented `AMI_SINGULAR_NORMALIZATION` error. AMI above 2**26
samples is rejected before allocation; the adjustable expectation-work budget
also rejects excessive work explicitly. String labels or safe integer Number
labels are supported, homogeneous within each input; other domains are rejected.

Stable definition compatibility is deliberately distinct from copying sklearn
rounding artifacts. For two different doublet partitions at N=3000, the exact
AMI is −2.2229637041155283e−7; JS differs by approximately 1.6e−16, while sklearn
1.8 returns −2.113112724214236e−6 (error 1.89e−6). For singleton-vs-doublet,
arithmetic AMI is exactly zero but sklearn can return about −3.78e−6. Undefined
optional-min results must not be compared as a normal finite-score parity case.
Full examples are in `audit/sklearn-conditioning.json`.

## Delivery and integration

Production artifact: `artifacts/ubukit-js-0.1.0-dev.4.tgz`.
Source/API contract: `package/src/external-metrics.js` and
`package/EXTERNAL_METRICS.md`. Retained attribution and BSD license are included.
The production patch is rooted at `preview/javascript/package` against dev3
GitHub commit `80caa7a0b90e1299d43cad18107290dc307dbcd4`; verification additions are
listed separately in `INTEGRATION_MANIFEST.json`. No GitHub, registry or public
site write is performed by this task. See `REPRODUCE.md` for all commands.
'''
(base/'REPORT.md').write_text(report)
print('Wrote report and matched summary')
