"""New explicit memory-policy wrapper; timed threaded.py remains unchanged.

max_scratch_bytes caps the modeled worker partials and block temporaries only.
It is NOT a total-RSS cap. Inputs, required outputs, full centered-coordinate
arrays, N-sized row vectors, Python objects and BLAS internal packing are outside
that cap and are reported separately where their sizes are predictable.
"""
import operator
from .oracle import check_inputs
from .threaded import run as _run


def workspace_plan(n,d,m,k,threads=1,block_rows=256,max_scratch_bytes=32<<20,distance='guarded'):
    n,d,m,k,threads,block_rows,max_scratch_bytes=map(operator.index,(n,d,m,k,threads,block_rows,max_scratch_bytes))
    if min(n,d,m,k,threads,block_rows,max_scratch_bytes)<1:
        raise ValueError('positive dimensions, threads, block size and scratch allowance required')
    if distance not in ('guarded','direct','centered'):
        raise ValueError('distance must be guarded, direct or centered')
    active=min(threads,n)
    partial_num=8*active*m*d
    partial_den=8*active*m
    reduction=8*(m*d+m)
    fixed=partial_num+partial_den+reduction
    # Guarded mode conservatively covers concurrent score scratch, up to three
    # additional B*M arrays during direct-row fallback, input gathers and row
    # vectors/indices. This deliberately exceeds the usual no-fallback path.
    per_worker_row=(8*(4*m+d+k+32)+9) if distance=='guarded' else 8*(m+16)
    all_workers_row=active*per_worker_row
    minimum=fixed+all_workers_row
    if max_scratch_bytes<minimum:
        raise MemoryError(f'scratch allowance {max_scratch_bytes} is below modeled one-row minimum {minimum}; reduce threads or increase allowance')
    requested=min(block_rows,(n+active-1)//active)
    chosen=min(requested,(max_scratch_bytes-fixed)//all_workers_row)
    score=8*active*chosen*m
    block_bound=all_workers_row*chosen
    centered=distance!='direct'
    centered_coordinates=8*(n*d+m*k+m*d+n*k) if centered else 0
    row_vectors=8*((4 if distance=='guarded' else 3)*n+3*m+d+k) if centered else 0
    outputs=8*(n*m+m*d+n*k)
    inputs=8*(n*d+m*k+m*d+n*m)
    return dict(requested_threads=threads,effective_threads=active,requested_block_rows=block_rows,
                chosen_block_rows=chosen,max_scratch_bytes=max_scratch_bytes,
                partial_numerator_bytes=partial_num,partial_denominator_bytes=partial_den,
                reduction_buffer_bytes=reduction,concurrent_score_buffer_bytes=score,
                conservative_block_temporary_bytes=block_bound,
                modeled_managed_bytes=fixed+block_bound,
                excluded_centered_coordinate_bytes=centered_coordinates,
                excluded_full_row_vector_bytes=row_vectors,
                required_output_bytes_excluding_history=outputs,input_bytes=inputs,
                budget_scope='worker partial accumulators + conservative block temporaries; not total RSS',
                excluded_unbounded_or_runtime_storage=['BLAS packing/workspace','Python object overhead','allocator retention','history storage','caller-owned references and validation copies'])


def run(X,R,W0,P0,gamma,lam,max_iters=100,tol=1e-4,threads=1,*,block_rows=256,
        max_scratch_bytes=32<<20,distance='guarded'):
    max_iters=operator.index(max_iters)
    check_inputs(X,R,W0,P0,gamma,lam,max_iters)
    if max_iters==0:
        out=_run(X,R,W0,P0,gamma,lam,0,tol,threads,block_rows=1,distance=distance)
        out['workspace_policy']=dict(no_iterations=True,modeled_managed_bytes=0,
                                     max_scratch_bytes=operator.index(max_scratch_bytes),
                                     budget_scope='no iterative worker allocation')
        return out
    plan=workspace_plan(X.shape[0],X.shape[1],R.shape[0],R.shape[1],threads,block_rows,max_scratch_bytes,distance)
    out=_run(X,R,W0,P0,gamma,lam,max_iters,tol,threads,
             block_rows=plan['chosen_block_rows'],distance=distance)
    out['workspace_policy']=plan
    out['variant']='policy_'+out['variant']
    return out
