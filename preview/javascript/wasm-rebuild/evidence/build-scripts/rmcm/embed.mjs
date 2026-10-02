import fs from 'node:fs';
const bytes=fs.readFileSync(new URL('./radius_candidates.wasm',import.meta.url));
const output=new URL('../javascript/rmcm-wasm.js',import.meta.url);
const source=fs.readFileSync(output,'utf8');
const updated=source.replace(/const bytes = new Uint8Array\(\[[^\]]*\]\);/,`const bytes = new Uint8Array([${[...bytes].join(', ')}]);`);
if(updated===source&&!source.includes(`new Uint8Array([${[...bytes].join(', ')}])`))throw new Error('embedded byte declaration not found');
fs.writeFileSync(output,updated);
console.log(`Embedded ${bytes.length} bytes`);
