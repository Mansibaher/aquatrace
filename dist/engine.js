export function scoreModel(trees, features){
  if(features.length!==6 || features.some(x=>!Number.isFinite(x)))throw Error('Expected six finite pressure residuals.');
  // scikit-learn casts prediction inputs to float32 before tree traversal.
  const input=features.map(Math.fround);
  return trees.reduce((sum,t)=>{let i=0;while(t.left[i]!==-1)i=input[t.feature[i]]<=t.threshold[i]?t.left[i]:t.right[i];return sum+t.value[i];},0)/trees.length;
}
function random(seed){return()=>{seed|=0;seed=seed+0x6D2B79F5|0;let t=Math.imul(seed^seed>>>15,1|seed);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
export function runExperiment(data,config){
  if(!data.sensors.includes(config.node)&&config.node!=='none')throw Error('Unknown sensor.');
  if(![0,1,2].includes(config.size)||![.7,1,1.3].includes(config.demand)||!Number.isFinite(config.noise)||config.noise<0||config.noise>.5||!Number.isInteger(config.seed))throw Error('Invalid experiment configuration.');
  const scenario=data.scenarios.find(s=>s.node===config.node&&s.demand===config.demand&&(config.node==='none'||s.size===config.size));
  if(!scenario)throw Error('Hydraulic scenario unavailable.');
  const rand=random(config.seed),normal=()=>Math.sqrt(-2*Math.log(Math.max(rand(),1e-12)))*Math.cos(2*Math.PI*rand());
  let streak=0,detectedAt=null,falseAlarms=0;
  const steps=Array.from({length:30},(_,i)=>{
    const leaking=config.node!=='none'&&i>=10;
    const pressure=(leaking?scenario.pressure:scenario.baseline).map(p=>p+normal()*config.noise);
    const residual=scenario.baseline.map((p,k)=>p-pressure[k]);
    const score=scoreModel(data.trees,residual),positive=score>=.5;
    if(!leaking&&positive)falseAlarms++;
    streak=leaking&&positive?streak+1:0;
    if(streak===3&&detectedAt===null)detectedAt=i;
    return {i,pressure,residual,score,positive,leaking};
  });
  return {config:{...config},scenario,steps,delay:detectedAt===null?null:detectedAt-10,falseAlarms};
}
