/** Strict float64 PCA Jacobi row rotations, with a per-fit owned workspace.
 * Built from wasm/pca_rotate.wat using wabt 1.0.39. No runtime download.
 */
const bytes=new Uint8Array([0,97,115,109,1,0,0,0,1,11,1,96,7,127,127,127,127,127,124,124,0,3,2,1,0,5,3,1,0,1,7,19,2,6,109,101,109,111,114,121,2,0,6,114,111,116,97,116,101,0,0,10,164,3,1,161,3,3,3,127,4,124,4,123,32,0,32,3,32,2,108,65,8,108,106,33,8,32,0,32,4,32,2,108,65,8,108,106,33,9,2,64,3,64,32,7,32,2,79,13,1,32,7,32,3,71,32,7,32,4,71,113,4,64,32,8,32,7,65,8,108,106,43,3,0,33,10,32,9,32,7,65,8,108,106,43,3,0,33,11,32,5,32,10,162,32,6,32,11,162,161,33,12,32,6,32,10,162,32,5,32,11,162,160,33,13,32,8,32,7,65,8,108,106,32,12,57,3,0,32,0,32,7,32,2,108,32,3,106,65,8,108,106,32,12,57,3,0,32,9,32,7,65,8,108,106,32,13,57,3,0,32,0,32,7,32,2,108,32,4,106,65,8,108,106,32,13,57,3,0,11,32,7,65,1,106,33,7,12,0,11,11,65,0,33,7,32,1,32,3,32,2,108,65,8,108,106,33,8,32,1,32,4,32,2,108,65,8,108,106,33,9,32,5,253,20,33,14,32,6,253,20,33,15,2,64,3,64,32,7,65,1,106,32,2,79,13,1,32,8,32,7,65,8,108,106,253,0,4,0,33,16,32,9,32,7,65,8,108,106,253,0,4,0,33,17,32,8,32,7,65,8,108,106,32,14,32,16,253,242,1,32,15,32,17,253,242,1,253,241,1,253,11,4,0,32,9,32,7,65,8,108,106,32,15,32,16,253,242,1,32,14,32,17,253,242,1,253,240,1,253,11,4,0,32,7,65,2,106,33,7,12,0,11,11,32,7,32,2,73,4,64,32,8,32,7,65,8,108,106,43,3,0,33,10,32,9,32,7,65,8,108,106,43,3,0,33,11,32,8,32,7,65,8,108,106,32,5,32,10,162,32,6,32,11,162,161,57,3,0,32,9,32,7,65,8,108,106,32,6,32,10,162,32,5,32,11,162,160,57,3,0,11,11]);
let compiled,unsupported=false;
export function pcaRotationWorkspace(dim,maxBytes,maxExtraBytes=Infinity) {
 if(!Number.isSafeInteger(dim) || dim<32 || unsupported || typeof WebAssembly==='undefined' || typeof WebAssembly.Module!=='function' || new Uint8Array(new Uint32Array([0x01020304]).buffer)[0]!==4)return null;
 const elements=dim*dim,required=16*elements,allocated=Math.ceil(required/65536)*65536;
 if(!Number.isSafeInteger(required) || required>=0x80000000 || allocated>maxBytes || allocated-required>maxExtraBytes)return null;
 try {
  if(!compiled)compiled=new WebAssembly.Module(bytes);
  const instance=new WebAssembly.Instance(compiled),memory=instance.exports.memory,pages=allocated/65536;
  if(pages>1)memory.grow(pages-1);
  const gram=new Float64Array(memory.buffer,0,elements),vectors=new Float64Array(memory.buffer,elements*8,elements);
  return {gram,vectors,allocatedBytes:allocated,rotate:(a,b,c,s)=>instance.exports.rotate(0,elements*8,dim,a,b,c,s)};
 } catch(error) {
  if(error instanceof WebAssembly.CompileError){unsupported=true;return null;}
  if(error instanceof RangeError || error instanceof TypeError || error?.name==='SecurityError' || error instanceof WebAssembly.LinkError)return null;
  throw error;
 }
}
