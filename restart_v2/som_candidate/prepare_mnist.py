"""Reproducible official MNIST input and common-initial-state fixture driver.

`inspect` verifies/downloads the small source archive and prints configuration.
`fixture` explicitly performs full SVD and, if requested, the reference run;
those operations must be scheduled in the parent's exclusive timing slot.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import numpy as np
from .initialization import initialize
from .oracle import run as oracle

URL='https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
SHA256='731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1'
DEFAULT_ARCHIVE=Path(__file__).resolve().parent/'data'/'mnist.npz'


def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def ensure_archive(path=DEFAULT_ARCHIVE):
    path=Path(path)
    if path.exists():
        actual=sha256_file(path)
        if actual!=SHA256:raise ValueError(f'MNIST SHA256 mismatch: {actual}')
        return path
    path.parent.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen(URL,timeout=120) as response:
        data=response.read()
    actual=hashlib.sha256(data).hexdigest()
    if actual!=SHA256:raise ValueError(f'Downloaded MNIST SHA256 mismatch: {actual}')
    path.write_bytes(data)
    return path


def make_grid(side=16):
    axis=np.linspace(-1.,1.,side,dtype=np.float64)
    gx,gy=np.meshgrid(axis,axis,indexing='xy')
    return np.ascontiguousarray(np.column_stack((gx.ravel(order='C'),gy.ravel(order='C'))))


def configuration(n=70000,side=16,gamma=2.767,lam=2.439,pca_scale=2.,initialization='direct'):
    R=make_grid(side)
    return dict(dataset='official Keras MNIST',source_url=URL,source_sha256=SHA256,
                concatenation='x_train followed by x_test; original archive order',
                selection=f'deterministic prefix of length {n}; no shuffle',random_seed=None,
                preprocessing='reshape to 784 pixels; float64 division by 255.0; no StandardScaler',
                n=int(n),d=784,m=int(side*side),grid_dimension=2,
                grid='meshgrid(linspace(-1,1,side), linspace(-1,1,side), indexing=xy); C ravel; columns x,y',
                grid_side=int(side),grid_sha256=hashlib.sha256(R.tobytes()).hexdigest(),
                gamma=float(gamma),lam=float(lam),pca_scale=float(pca_scale),
                initialization=initialization,
                svd='numpy.linalg.svd(X-X.mean(0), full_matrices=False); full SVD',
                earlier_measurements_reused=False)


def load_mnist(n=70000,archive=DEFAULT_ARCHIVE):
    n=int(n)
    if not 1<=n<=70000:raise ValueError('n must be in [1,70000]')
    path=ensure_archive(archive)
    with np.load(path,allow_pickle=False) as z:
        train=z['x_train'];test=z['x_test'];yt=z['y_train'];yv=z['y_test']
        if train.shape!=(60000,28,28) or test.shape!=(10000,28,28):
            raise ValueError('unexpected official MNIST array shape')
        if train.dtype!=np.uint8 or test.dtype!=np.uint8:
            raise ValueError('unexpected pixel dtype')
        if n<=60000:
            pixels=train[:n];labels=yt[:n].copy()
        else:
            pixels=np.concatenate((train,test[:n-60000]),axis=0)
            labels=np.concatenate((yt,yv[:n-60000]))
        X=pixels.reshape(n,784).astype(np.float64)
        X/=255.
    return np.ascontiguousarray(X),labels


def create_fixture(path,n=2000,side=16,gamma=2.767,lam=2.439,pca_scale=2.,
                   initialization='direct',threads=1,reference_iters=None,reference_tol=-1.,
                   archive=DEFAULT_ARCHIVE):
    X,labels=load_mnist(n,archive);R=make_grid(side)
    meta=configuration(n,side,gamma,lam,pca_scale,initialization)
    started=time.perf_counter()
    W0,P0=initialize(X,R,lam,pca_scale,threads,method=initialization)
    meta['initialization_seconds']=time.perf_counter()-started
    meta['initialization_threads']=int(threads)
    for name,value in [('X',X),('W0',W0),('P0',P0)]:
        meta[name+'_sha256']=hashlib.sha256(value.tobytes()).hexdigest()
    values=dict(X=X,R=R,W0=W0,P0=P0,labels=labels,gamma=np.asarray(gamma),lam=np.asarray(lam))
    if reference_iters is not None:
        started=time.perf_counter()
        ref=oracle(X,R,W0,P0,gamma,lam,reference_iters,reference_tol,threads)
        meta.update(reference_seconds=time.perf_counter()-started,reference_iters=int(reference_iters),
                    reference_tol=float(reference_tol),reference_threads=int(threads))
        for key in ['W','P','V','history','n_iter']:values['ref_'+key]=ref[key]
    values['metadata']=np.asarray(json.dumps(meta,sort_keys=True))
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);np.savez(path,**values)
    path.with_suffix('.metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['inspect','fixture'])
    parser.add_argument('--archive',default=str(DEFAULT_ARCHIVE))
    parser.add_argument('--n',type=int,default=2000);parser.add_argument('--side',type=int,default=16)
    parser.add_argument('--gamma',type=float,default=2.767);parser.add_argument('--lam',type=float,default=2.439)
    parser.add_argument('--pca-scale',type=float,default=2.)
    parser.add_argument('--initialization',choices=['direct','same_svd_lowrank'],default='direct')
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--reference-iters',type=int);parser.add_argument('--reference-tol',type=float,default=-1.)
    parser.add_argument('--output')
    a=parser.parse_args();ensure_archive(a.archive)
    if a.action=='inspect':
        result=configuration(a.n,a.side,a.gamma,a.lam,a.pca_scale,a.initialization)
        result['archive_bytes']=Path(a.archive).stat().st_size
        result['fixture_created']=False;result['svd_or_reference_executed']=False
    else:
        if not a.output:parser.error('fixture requires --output')
        result=create_fixture(a.output,a.n,a.side,a.gamma,a.lam,a.pca_scale,a.initialization,
                              a.threads,a.reference_iters,a.reference_tol,a.archive)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
