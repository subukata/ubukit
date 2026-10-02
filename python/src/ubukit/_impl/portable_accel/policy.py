"""Scoped thread policy; named scratch limits do not bound total RSS."""
from contextlib import contextmanager,nullcontext
from dataclasses import dataclass
from .validation import positive_int
@dataclass(frozen=True)
class ExecutionPolicy:
    threads:int|None=1
    block_rows:int=256
    max_scratch_bytes:int=32*2**20
    def __post_init__(self):
        if self.threads is not None:positive_int(self.threads,'threads')
        positive_int(self.block_rows,'block_rows');positive_int(self.max_scratch_bytes,'max_scratch_bytes')
    def rows_for(self,bytes_per_row,rows):
        positive_int(bytes_per_row,'bytes_per_row');positive_int(rows,'rows')
        limit=self.max_scratch_bytes//bytes_per_row
        if not limit:raise ValueError('scratch cap cannot hold one row')
        return min(self.block_rows,rows,limit)
    @contextmanager
    def activate(self,*,numba=False,blas=True):
        from ._threadpools import threadpool_context
        old=None
        if numba:
            from numba import get_num_threads,set_num_threads
            old=get_num_threads()
        try:
            if numba and self.threads is not None:set_num_threads(self.threads)
            context=nullcontext() if self.threads is None else threadpool_context(limits=self.threads,user_api=None if blas else 'openmp')
            with context:yield self
        finally:
            if numba and old is not None:set_num_threads(old)
def policy_or_default(policy):
    if policy is None:return ExecutionPolicy()
    if not isinstance(policy,ExecutionPolicy):raise TypeError('policy must be ExecutionPolicy')
    return policy
