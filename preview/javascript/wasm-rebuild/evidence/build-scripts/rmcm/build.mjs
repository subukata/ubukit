import wabtFactory from 'wabt';
import fs from 'node:fs';
const wabt = await wabtFactory();
const source=fs.readFileSync(new URL('./radius_candidates.wat',import.meta.url),'utf8');
const parsed = wabt.parseWat('radius_candidates.wat',source,{simd:true});
parsed.resolveNames();parsed.validate({simd:true});
const {buffer}=parsed.toBinary({log:false,write_debug_names:true});
fs.writeFileSync(new URL('./radius_candidates.wasm',import.meta.url),buffer);
console.log(buffer.length,WebAssembly.validate(buffer));
