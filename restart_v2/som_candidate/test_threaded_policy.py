"""Tiny new-wrapper checks and allocation arithmetic only; no large arrays."""
import json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from .threaded_policy import workspace_plan,run
from .oracle import run as reference

def main():
    plans=[workspace_plan(70000,784,256,2,9,256,budget) for budget in [32<<20,64<<20]]
    assert all(p['modeled_managed_bytes']<=p['max_scratch_bytes'] for p in plans)
    assert plans[1]['chosen_block_rows']==256
    assert plans[0]['chosen_block_rows']<256
    try:workspace_plan(70000,784,256,2,9,256,1024)
    except MemoryError:pass
    else:raise AssertionError('impossible allowance accepted')
    rng=np.random.default_rng(883);X=rng.normal(size=(23,5));R=rng.normal(size=(13,2));W=rng.normal(size=(13,5));P=softmax(rng.normal(size=(23,13)),axis=1)
    worst=0.;count=0
    for distance in ['guarded','direct']:
        for budget in [4096,32768]:
            ref=reference(X,R,W,P,1.2,.5,7,-1,1)
            out=run(X,R,W,P,1.2,.5,7,-1,2,max_scratch_bytes=budget,distance=distance)
            assert out['workspace_policy']['modeled_managed_bytes']<=budget
            for key in ['W','P','V','history']:
                worst=max(worst,float(np.max(np.abs(ref[key]-out[key]))))
                np.testing.assert_allclose(out[key],ref[key],rtol=1e-9,atol=1e-9)
            count+=1
    result=dict(status='pass',tiny_cases=count,worst_absolute=worst,mnist70k_plans=plans,
                disclaimer='Arithmetic estimates only; no total-RSS guarantee or new performance measurements')
    Path(__file__).with_name('threaded_policy_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()
