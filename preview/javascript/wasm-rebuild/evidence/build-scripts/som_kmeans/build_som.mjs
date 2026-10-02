import wabtFactory from '../../../../acceleration_round3/work/rmcm/wasm/node_modules/wabt/index.js';
import fs from'node:fs';
const wabt=await wabtFactory();const source=fs.readFileSync(new URL('som_train.wat',import.meta.url),'utf8');
const module=wabt.parseWat('som_train.wat',source,{simd:true});module.resolveNames();module.validate({simd:true});
const{buffer}=module.toBinary({write_debug_names:false});fs.writeFileSync(new URL('som_train.wasm',import.meta.url),buffer);
fs.writeFileSync(new URL('som_train_bytes.json',import.meta.url),JSON.stringify([...buffer]));console.log(buffer.length,WebAssembly.validate(buffer));
