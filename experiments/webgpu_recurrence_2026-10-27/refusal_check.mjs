// The forms the WebGPU path refuses (irSupport: the worker then serves the document in WebAssembly, with a console
// warning naming the reason), on variants of pk_recurrence's document; no GPU (irSupport compiles the bodies only).
//   node refusal_check.mjs PK_RECURRENCE_DOC.json OUT.json
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const { irSupport, REC_SUM_NEST } = await import(pathToFileURL(path.resolve(here, '../../deploy/wasm/web/webgpu/ir_webgpu.js')).href);
const base = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const recIdx = base.block.nodes.findIndex((n) => n.op === 'recurrence');
const variant = (f) => { const d = structuredClone(base); f(d, d.block.nodes[recIdx]); return d; };
const cases = {
  as_exported: variant(() => {}),
  deepest_sum_nesting_that_parses: null,   // filled below: the body depth limit (6.10.2) bounds nested sums
  outside_the_block: variant((d, n) => { d.head.nodes.unshift(d.block.nodes.splice(recIdx, 1)[0]); }),
  beside_depth_route: variant((d) => { d.block.nodes.unshift({ op: 'depth_route', in: ['x'], weight: 'router.weight', n_rec: 2, step: 0 }); }),
  body_does_not_parse: variant((d, n) => { n.readout = 'y[h,e] = num[h,e] +'; }),
  row_reduction_in_a_body: variant((d, n) => { n.readout = 'y[h,e] = rowsum(num[h,e])'; }),
  index_out_of_scope: variant((d, n) => { n.let[0] = 'pk[h,e] = k[h,f]'; }),
};
const nested = (k) => variant((d, n) => {   // k nested sums over fresh indices of size 1
  for (let j = 0; j < k; j++) n.index[`z${j}`] = 1;
  let body = 'num[h,e]';
  for (let j = k - 1; j >= 0; j--) body = `sum(z${j}, ${body})`;
  n.update[0] = `num[h,e] = 0.9375 * ${body} + pk[h,e] * v[h,e]`;
});
let kmax = 0;
while (irSupport(nested(kmax + 1)).ok) kmax++;
cases.deepest_sum_nesting_that_parses = nested(kmax);
cases.one_sum_deeper = nested(kmax + 1);
const out = { deepest_sum_nesting: kmax };
for (const [k, d] of Object.entries(cases)) { out[k] = irSupport(d); console.log(k.padEnd(28), JSON.stringify(out[k])); }
fs.writeFileSync(process.argv[3], JSON.stringify({ REC_SUM_NEST, cases: out }, null, 1) + '\n');
