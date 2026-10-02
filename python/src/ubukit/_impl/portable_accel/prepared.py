"""Owned immutable snapshots. New implementation, with explicit cache keys."""
import numpy as np
from .validation import matrix

def _immutable(x):
    x=np.ascontiguousarray(x)
    return np.frombuffer(x.tobytes(),dtype=x.dtype).reshape(x.shape)
class PreparedData:
    __slots__=('_X','_cache','_hits','_misses')
    def __init__(self,X,*,dtype=None):
        self._X=_immutable(matrix(X,dtype=dtype));self._cache={};self._hits=self._misses=0
    @property
    def X(self):return self._X
    def _get(self,preprocessing,contract,operation,make):
        key=(self.X.dtype.str,preprocessing,contract,operation,np.__version__)
        if key in self._cache:self._hits+=1;return self._cache[key]
        value=make();self._cache[key]=value;self._misses+=1;return value
    def as_dtype(self,dtype):
        dtype=np.dtype(dtype)
        if dtype==self.X.dtype:return self
        return self._get('original','finite-cast',dtype.str,lambda:PreparedData(self.X,dtype=dtype))
    def feature_bounds(self):
        return self._get('original','numpy-axis0-minmax','bounds',lambda:(_immutable(self.X.min(axis=0)),_immutable(self.X.max(axis=0))))
    def centered(self):
        return self._get('mean-centered','numpy-axis0-mean','centered',lambda:_immutable(self.X-self.X.mean(axis=0)))
    def norms(self,*,preprocessing='original',contract='squared-euclidean'):
        if preprocessing not in ('original','mean-centered') or contract!='squared-euclidean':raise ValueError('unsupported norm contract')
        x=self.X if preprocessing=='original' else self.centered()
        return self._get(preprocessing,contract,'numpy-einsum',lambda:_immutable(np.einsum('ij,ij->i',x,x)))
    def cache_info(self):return {'hits':self._hits,'misses':self._misses,'keys':tuple(self._cache)}
    def clear_cache(self):self._cache.clear();self._hits=self._misses=0
def prepare(X,*,dtype=None):
    if isinstance(X,PreparedData):return X if dtype is None else X.as_dtype(dtype)
    return PreparedData(X,dtype=dtype)
