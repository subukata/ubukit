"""New restart-v2 validation layer."""
from numbers import Integral
import numpy as np

def positive_int(value,name,*,allow_zero=False):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,Integral) or value<(0 if allow_zero else 1):raise ValueError(name+' must be a valid integer')
    return int(value)

def matrix(value,*,dtype=None,name='X'):
    x=np.asarray(value)
    if x.ndim!=2 or not all(x.shape):raise ValueError(name+' must be nonempty and 2D')
    if x.dtype.kind not in 'fiu':raise TypeError(name+' must be real')
    if dtype is None:dtype=np.float32 if x.dtype==np.float32 else np.float64
    with np.errstate(over='ignore',invalid='ignore'):x=np.ascontiguousarray(x,dtype=dtype)
    for i in range(0,len(x),256):
        if not np.isfinite(x[i:i+256]).all():raise ValueError(name+' must be finite')
    return x

def matrix_input(value,*,dtype=None,name='X'):
    from .prepared import PreparedData
    if isinstance(value,PreparedData):
        p=value if dtype is None or value.X.dtype==np.dtype(dtype) else value.as_dtype(dtype)
        return p.X,p
    return matrix(value,dtype=dtype,name=name),None
