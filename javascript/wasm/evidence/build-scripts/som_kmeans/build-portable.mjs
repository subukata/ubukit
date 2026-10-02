import wabtFactory from'wabt';import path from'node:path';import{build}from'./build-common.mjs';
const at=process.argv.indexOf('--output-root');console.log(JSON.stringify(await build(wabtFactory,{check:process.argv.includes('--check'),outputRoot:at<0?null:path.resolve(process.argv[at+1])}),null,2));
