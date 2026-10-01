"""Compare literal, scaled-power, all-log and specialized/hybrid thresholds.

This measures only membership assignment from precomputed squared distances.
It is not an end-to-end fit speed claim. All raw timings and correctness flags
are retained, including intentionally unsafe literal-overflow examples.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import numpy as np
from rough_cmeans import _mask_from_sq, _scaled_power_mask


def log_mask(sq, alpha, beta, p):
    d = np.sqrt(sq)
    with np.errstate(all='ignore'):
        a = alpha * d.min(axis=1)
        if beta == 0:
            return d <= a[:, None]
        scale = np.maximum(a, beta)
        small = np.minimum(a, beta)
        relative = (d - scale[:, None]) / scale[:, None]
        log_left = np.log1p(relative)
        fallback = np.isinf(relative)
        alt = np.log(d) - np.log(scale[:, None])
        log_left[fallback] = alt[fallback]
        ratio = small / scale
        log_small = np.log(ratio)
        tiny = ratio < np.finfo(float).tiny
        log_small[tiny] = np.log(small[tiny]) - np.log(scale[tiny])
        right = np.logaddexp(0, p * log_small)
        M = p * log_left <= right[:, None]
        M |= d <= scale[:, None]
        M[small == 0] = d[small == 0] <= scale[small == 0, None]
        M[np.isinf(scale)] = True
    return M


def literal_mask(sq, alpha, beta, p):
    d = np.sqrt(sq)
    with np.errstate(all='ignore'):
        return d ** np.float64(p) <= np.float64(alpha) ** p * d.min(axis=1)[:, None] ** p + np.float64(beta) ** p


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', default='results/threshold_ablation.json'); ap.add_argument('--repeats', type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(505)
    sq = rng.uniform(1, 100, size=(8192, 16))
    rows=[]
    cases=[('ordinary', sq, 1.15, .5, p) for p in [1.,2.,3.,.001,1000.]]
    edge=np.tile(np.array([[1e20,1e308]]), (1024,1))
    cases += [('tiny_beta_fractional_p',edge,1.,1e-320,.001)]
    funcs={'literal':literal_mask, 'scaled_power_with_guards':_scaled_power_mask, 'all_log':log_mask, 'specialized_hybrid':_mask_from_sq}
    for name, matrix, alpha,beta,p in cases:
        oracle=log_mask(matrix,alpha,beta,p)
        raw={key:[] for key in funcs}
        outputs={key:f(matrix,alpha,beta,p) for key,f in funcs.items()}
        names=list(funcs)
        for repeat in range(args.repeats):
            order=names[repeat%len(names):]+names[:repeat%len(names)]
            for key in order:
                begin=time.perf_counter(); funcs[key](matrix,alpha,beta,p); raw[key].append(time.perf_counter()-begin)
        for key in funcs:
            rows.append(dict(case=name,n=matrix.shape[0],k=matrix.shape[1],alpha=alpha,beta=beta,p=p,method=key,seconds=raw[key],median_seconds=statistics.median(raw[key]),mask_mismatches=int(np.count_nonzero(outputs[key]!=oracle)),correctness_pass=bool(np.array_equal(outputs[key],oracle))))
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    hashes={f:hashlib.sha256((Path(__file__).parent/f).read_bytes()).hexdigest() for f in ['rough_cmeans.py','threshold_ablation.py']}
    out.write_text(json.dumps(dict(source_sha256=hashes,seed=505,note='Threshold-only; full distances precomputed. All-log oracle plus independent edge-case tests; literal overflow mismatches are deliberately reported, not silently accepted.',results=rows),indent=2))
    for row in rows: print(json.dumps(row))


if __name__=='__main__': main()
