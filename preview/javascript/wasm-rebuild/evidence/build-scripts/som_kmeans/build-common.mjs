import fs from'node:fs';import path from'node:path';import{fileURLToPath}from'node:url';
export async function build(wabtFactory,{check=false,outputRoot=null}={}){
 const here=path.dirname(fileURLToPath(import.meta.url)),root=outputRoot??path.resolve(here,'../javascript'),wabt=await wabtFactory(),records=[];
 for(const[wat,javascript]of[['pca_rotate.wat','som-pca-wasm.js'],['kmeans_r2w8.wat','kmeans-wasm.js'],['som_train.wat','som-training-wasm.js']]){
  const source=fs.readFileSync(path.join(here,wat),'utf8'),module=wabt.parseWat(wat,source,{simd:true});module.resolveNames();module.validate({simd:true});const{buffer}=module.toBinary({write_debug_names:false});if(!WebAssembly.validate(buffer))throw Error(wat+' did not validate');
  const target=path.join(root,javascript),text=fs.readFileSync(target,'utf8'),pattern=/const bytes\s*=\s*new Uint8Array\((\[[\s\S]*?\])\);/,match=text.match(pattern);if(!match)throw Error('Embedded byte array not found in '+target);
  const equal=Buffer.from(JSON.parse(match[1])).equals(Buffer.from(buffer));if(check&&!equal)throw Error('Embedded bytes differ: '+javascript);
  if(!check)fs.writeFileSync(target,text.replace(pattern,'const bytes=new Uint8Array('+JSON.stringify([...buffer])+');'));
  records.push({wat,javascript,bytes:buffer.length,embedded_matches:equal});module.destroy();
 }
 return records;
}
