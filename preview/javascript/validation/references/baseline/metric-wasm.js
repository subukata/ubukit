/** Optional embedded WebAssembly SIMD. No runtime fetch or npm dependency.
 * Built from the accompanying distance_rows8.wat with wabt 1.0.39.
 * A per-call transposed input snapshot is budgeted including memory pages.
 * Keep inputs stable throughout generator execution for backend equivalence:
 * the historical JavaScript path reads its caller-owned Float64 arrays.
 */
const bytes = new Uint8Array([0,97,115,109,1,0,0,0,1,26,3,96,5,127,127,127,127,127,1,127,96,3,127,127,127,1,127,96,6,127,127,127,127,127,127,0,3,4,3,0,1,2,5,3,1,0,1,7,30,4,6,109,101,109,111,114,121,2,0,3,114,111,119,0,0,4,114,97,110,107,0,1,4,104,105,115,116,0,2,10,140,9,3,130,6,4,3,127,2,123,2,124,5,123,65,0,33,5,2,64,3,64,32,5,65,7,106,32,2,79,13,1,65,0,33,6,253,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,33,12,253,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,33,13,253,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,33,14,253,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,33,15,2,64,3,64,32,6,32,3,79,13,1,32,0,32,6,32,2,108,65,8,108,106,33,7,32,7,32,1,65,8,108,106,43,3,0,253,20,33,16,32,16,32,7,32,5,65,8,108,106,253,0,3,0,253,241,1,33,9,32,12,32,9,32,9,253,242,1,253,240,1,33,12,32,16,32,7,32,5,65,8,108,106,253,0,3,16,253,241,1,33,9,32,13,32,9,32,9,253,242,1,253,240,1,33,13,32,16,32,7,32,5,65,8,108,106,253,0,3,32,253,241,1,33,9,32,14,32,9,32,9,253,242,1,253,240,1,33,14,32,16,32,7,32,5,65,8,108,106,253,0,3,48,253,241,1,33,9,32,15,32,9,32,9,253,242,1,253,240,1,33,15,32,6,65,1,106,33,6,12,0,11,11,32,12,68,255,255,255,255,255,255,239,127,253,20,253,75,253,195,1,69,4,64,65,1,15,11,32,4,32,5,65,8,108,106,32,12,253,239,1,253,11,3,0,32,13,68,255,255,255,255,255,255,239,127,253,20,253,75,253,195,1,69,4,64,65,1,15,11,32,4,32,5,65,8,108,106,32,13,253,239,1,253,11,3,16,32,14,68,255,255,255,255,255,255,239,127,253,20,253,75,253,195,1,69,4,64,65,1,15,11,32,4,32,5,65,8,108,106,32,14,253,239,1,253,11,3,32,32,15,68,255,255,255,255,255,255,239,127,253,20,253,75,253,195,1,69,4,64,65,1,15,11,32,4,32,5,65,8,108,106,32,15,253,239,1,253,11,3,48,32,5,65,8,106,33,5,12,0,11,11,2,64,3,64,32,5,65,1,106,32,2,79,13,1,65,0,33,6,253,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,33,8,2,64,3,64,32,6,32,3,79,13,1,32,0,32,6,32,2,108,65,8,108,106,33,7,32,7,32,1,65,8,108,106,43,3,0,253,20,32,7,32,5,65,8,108,106,253,0,3,0,253,241,1,33,9,32,8,32,9,32,9,253,242,1,253,240,1,33,8,32,6,65,1,106,33,6,12,0,11,11,32,8,68,255,255,255,255,255,255,239,127,253,20,253,75,253,195,1,69,4,64,65,1,15,11,32,4,32,5,65,8,108,106,32,8,253,239,1,253,11,3,0,32,5,65,2,106,33,5,12,0,11,11,32,5,32,2,73,4,64,65,0,33,6,68,0,0,0,0,0,0,0,0,33,10,2,64,3,64,32,6,32,3,79,13,1,32,0,32,6,32,2,108,65,8,108,106,33,7,32,7,32,1,65,8,108,106,43,3,0,32,7,32,5,65,8,108,106,43,3,0,161,33,11,32,10,32,11,32,11,162,160,33,10,32,6,65,1,106,33,6,12,0,11,11,32,10,68,255,255,255,255,255,255,239,127,100,4,64,65,1,15,11,32,4,32,5,65,8,108,106,32,10,159,57,3,0,11,65,0,11,185,1,3,2,127,2,124,3,123,32,0,32,2,65,8,108,106,43,3,0,33,5,65,1,33,4,2,64,3,64,32,3,65,1,106,32,1,79,13,1,32,0,32,3,65,8,108,106,253,0,3,0,33,7,32,3,173,253,18,253,12,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,253,206,1,33,8,32,7,32,5,253,20,253,73,32,7,32,5,253,20,253,71,32,8,32,2,173,253,18,253,216,1,253,78,253,80,33,9,32,4,32,9,253,196,1,105,106,33,4,32,3,65,2,106,33,3,12,0,11,11,32,3,32,1,73,4,64,32,0,32,3,65,8,108,106,43,3,0,33,6,32,6,32,5,99,32,6,32,5,97,32,3,32,2,73,113,114,4,64,32,4,65,1,106,33,4,11,11,32,4,11,202,1,2,6,127,2,124,2,64,3,64,32,6,32,4,75,13,1,32,5,32,6,65,4,108,106,65,0,54,2,0,32,6,65,1,106,33,6,12,0,11,11,65,0,33,6,2,64,3,64,32,6,32,1,79,13,1,32,6,32,2,71,4,64,32,0,32,6,65,8,108,106,43,3,0,33,12,65,0,33,7,32,4,33,8,2,64,3,64,32,7,32,8,79,13,1,32,7,32,8,106,65,1,118,33,9,32,3,32,9,65,4,108,106,40,2,0,33,10,32,0,32,10,65,8,108,106,43,3,0,33,13,32,13,32,12,99,32,13,32,12,97,32,10,32,6,73,113,114,4,64,32,9,65,1,106,33,7,5,32,9,33,8,11,12,0,11,11,32,5,32,7,65,4,108,106,33,11,32,11,32,11,40,2,0,65,1,106,54,2,0,11,32,6,65,1,106,33,6,12,0,11,11,11]);
let compiled, unsupported = false;
export function wasmRows(x,y,extraBudget,maxK) {
  if(unsupported || typeof WebAssembly==='undefined' || typeof WebAssembly.Module!=='function' || new Uint8Array(new Uint32Array([0x01020304]).buffer)[0]!==4) return null;
  const n=x.nSamples, xBytes=x.data.length*8, yBytes=y.data.length*8;
  const orderOffset=xBytes+yBytes+16*n, histOffset=orderOffset+4*maxK;
  const bytesNeeded=histOffset+4*(maxK+1);
  const allocatedBytes=Math.ceil(bytesNeeded/65536)*65536, extraBytes=allocatedBytes-16*n;
  if(extraBytes>extraBudget || bytesNeeded>=0x80000000) return null;
  try {
    if(!compiled){try {compiled = new WebAssembly.Module(bytes);}catch(error){if(error instanceof WebAssembly.CompileError){unsupported=true;return null;}throw error;}}
    const instance=new WebAssembly.Instance(compiled);
    const memory=instance.exports.memory;
    const pages=Math.ceil(bytesNeeded/65536);
    if(pages>1) memory.grow(pages-1);
    const all=new Float64Array(memory.buffer);
    for(let f=0;f<x.nFeatures;f++)for(let i=0;i<n;i++)all[f*n+i]=x.data[i*x.nFeatures+f];
    for(let f=0;f<y.nFeatures;f++)for(let i=0;i<n;i++)all[x.data.length+f*n+i]=y.data[i*y.nFeatures+f];
    const dx=new Float64Array(memory.buffer,xBytes+yBytes,n),dy=new Float64Array(memory.buffer,xBytes+yBytes+8*n,n);
    const orders=new Uint32Array(memory.buffer,orderOffset,maxK),counts=new Uint32Array(memory.buffer,histOffset,maxK+1);
    const histogram=(base,order,k,hist,self)=>{orders.set(order.subarray(0,k));instance.exports.hist(base,n,self,orderOffset,k,histOffset);for(let a=0;a<=k;a++)hist[a]=counts[a];};
    return {dx,dy,extraBytes,histX:(order,k,hist,self)=>histogram(xBytes+yBytes,order,k,hist,self),histY:(order,k,hist,self)=>histogram(xBytes+yBytes+8*n,order,k,hist,self),rankX(q){return instance.exports.rank(xBytes+yBytes,n,q)},rankY(q){return instance.exports.rank(xBytes+yBytes+8*n,n,q)},row(i){
      if(instance.exports.row(0,i,n,x.nFeatures,xBytes+yBytes)||instance.exports.row(xBytes,i,n,y.nFeatures,xBytes+yBytes+8*n))throw new RangeError('Euclidean squared distance overflowed; rescale the input');
      dx[i]=Infinity;dy[i]=Infinity;
    }};
  } catch(error) { if(error instanceof RangeError || error instanceof TypeError || error?.name==='SecurityError' || error instanceof WebAssembly.CompileError || error instanceof WebAssembly.LinkError) return null; throw error; }
}
