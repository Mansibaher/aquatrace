import {readFileSync} from 'node:fs';
import assert from 'node:assert/strict';
import {scoreModel,runExperiment} from '../dist/engine.js';
const data=JSON.parse(readFileSync(new URL('../dist/data.json',import.meta.url)));
const parity=JSON.parse(readFileSync(new URL('./parity.json',import.meta.url)));
for(let i=0;i<parity.features.length;i++)assert.ok(Math.abs(scoreModel(data.trees,parity.features[i])-parity.scores[i])<1e-10,'Exported inference must match sklearn');
let runs=0;
for(const s of data.scenarios)for(const noise of [0,.1,.5]){
 const config={node:s.node,size:s.size,demand:s.demand,noise,seed:42};
 const r=runExperiment(data,config);assert.equal(r.steps.length,30);assert.deepEqual(r,runExperiment(data,config));
 assert.ok(r.steps.every(x=>x.pressure.every(Number.isFinite)&&x.score>=0&&x.score<=1));
 assert.ok(r.steps.slice(0,10).every(x=>!x.leaking));
 if(s.node==='none')assert.ok(r.steps.every(x=>!x.leaking));
 if(noise===0){assert.deepEqual(r.steps[0].pressure,s.baseline);assert.deepEqual(r.steps[29].pressure,s.pressure);}
 runs++;
}
assert.throws(()=>runExperiment(data,{node:'fake',size:1,demand:1,noise:.1,seed:42}));
assert.throws(()=>runExperiment(data,{node:'none',size:1,demand:1,noise:-1,seed:42}));
console.log(`PASS: 100 sklearn parity fixtures, ${runs} scenarios/noise combinations, determinism, timing, invalid inputs.`);
