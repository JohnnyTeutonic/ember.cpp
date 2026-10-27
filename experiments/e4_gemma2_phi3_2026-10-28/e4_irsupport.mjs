// E4: the page's WebGPU support check (deploy/wasm/web/webgpu/ir_webgpu.js irSupport) on IR documents, run in
// node without a GPU adapter (irSupport reads the document only). A document it refuses stays in WebAssembly.
//   node e4_irsupport.mjs IR_WEBGPU_JS DOC.json ...
import { readFileSync } from 'fs';
import { pathToFileURL } from 'url';

const [mod, ...docs] = process.argv.slice(2);
const { irSupport } = await import(pathToFileURL(mod).href);
for (const p of docs) {
  const doc = JSON.parse(readFileSync(p, 'utf8'));
  console.log(JSON.stringify({ doc: p, ...irSupport(doc) }));
}
