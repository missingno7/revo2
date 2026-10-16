'use strict';
const $=id=>document.getElementById(id);
let state,revision=-1,initialized=false,controlsKey='',toastTimer;
const inputs=['x','y','radius','sin(πx)','sin(πy)','1'];
const ref=i=>inputs[i]??`v${i}`;
const integer=n=>n.toLocaleString('en-US');
function toast(message){$('toast').textContent=message;$('toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').hidden=true,3500);}
async function action(action,extra={}){
  try{const response=await fetch('/api/action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action,...extra})});if(!response.ok)throw new Error(await response.text());}
  catch(error){toast(error.message);}
}
function formProblem(kind,preset){
  const packing=kind==='packing';
  $('nodesField').hidden=packing;$('rectangleField').hidden=!packing;$('rectangleSeedField').hidden=!packing;
  $('uploadLabel').hidden=packing;$('presetLabel').textContent=packing?'Rectangle set':'Target image';
  const options=packing?[['mixed','Mixed rectangles'],['tiles','Modular tiles'],['bars','Long rectangles']]:[['orbit','Orbit'],['rings','Chromatic rings'],['sunset','Sunset'],['waves','Interference']];
  $('preset').replaceChildren(...options.map(([value,text])=>Object.assign(document.createElement('option'),{value,textContent:text})));
  if(preset&&!options.some(([value])=>value===preset))$('preset').append(Object.assign(document.createElement('option'),{value:packing?'custom':'',textContent:packing?'Custom rectangles':'Uploaded image'}));
  $('preset').value=options.some(([value])=>value===preset)?preset:(preset?(packing?'custom':''):options[0][0]);
}
function syncControls(s){
  $('problem').value=s.config.problem;formProblem(s.config.problem,s.preset);
  $('populationWidth').value=s.config.width;$('populationHeight').value=s.config.height;$('nodes').value=s.config.nodes;
  $('rectangleCount').value=s.config.rectangle_count;$('rectangleSeed').value=s.config.rectangle_seed;
  $('mutation').value=Math.round(s.config.mutation_rate*100);$('mutationValue').textContent=$('mutation').value+'%';
  $('seed').value=s.config.seed;$('backend').value=s.backend_key;
}
function presentation(packing){
  $('heroTitle').replaceChildren(document.createTextNode(packing?'Less space.':'Programs'),document.createElement('br'),document.createTextNode(packing?'Better fit':'become pictures'),Object.assign(document.createElement('span'),{textContent:'.'}));
  $('heroDescription').textContent=packing?'Fixed rectangles. Evolving order and rotations. Find a smaller footprint.':'A population of tiny programs. One image to evolve toward.';
  $('uploadNote').hidden=packing;$('inspectorTitle').textContent=packing?'Layout inspector':'Program inspector';
  $('legendHigh').textContent=packing?'More waste':'Higher error';$('legendLow').textContent=packing?'Less waste':'Lower error';
  $('chart').setAttribute('aria-label',packing?'Best and mean fraction of unused bounding area over generations':'Best and mean penalized image error over generations');
  $('selectedImage').alt=packing?'Selected rectangle packing layout':'Selected image program';
  $('program').setAttribute('aria-label',packing?'Selected rectangle placement order':'Selected program instructions');
  $('chartNote').textContent=packing?'Fraction of unused area · logarithmic scale':'Lower is better · logarithmic error scale';
  $('canvasTitle').textContent=packing?'The packing floor':'Evolution canvas';
  $('scoreLabel').textContent=packing?'BOUNDING AREA':'TRAINING ERROR';
  $('referenceTitle').textContent=packing?'Rectangle inventory':'Target';$('referenceNote').textContent=packing?'Every piece must fit, without overlap':'The image we’re searching for';
  $('bestTitle').textContent=packing?'Densest layout':'Best approximation';$('solutionTag').textContent=packing?'90° ROTATION ALLOWED':'EVOLVING PROGRAM';
  $('inspectorMetricLabel').textContent=packing?'FILL RATIO':'ACTIVE NODES';$('inspectionTitle').textContent=packing?'PLACEMENT ORDER':'EXPRESSION GRAPH';
  $('inspectionHint').textContent=packing?'DIMENSIONS · X, Y':'RGB ↙ LAST 3 NODES';
  $('inspectionNote').textContent=packing?'Each numbered piece retains its dimensions. ↻ marks a 90° rotation. Placement uses a skyline decoder; evolution searches order, orientation and width.':'Muted instructions are inactive. Inputs are x, y, radius, two sine waves, and 1.';
  $('textExport').href=packing?'/layout.svg':'/program.glsl';$('textExport').download=packing?'revo-layout.svg':'revo-program.glsl';$('textExport').textContent=packing?'Export best SVG ↗':'Export best GLSL ↗';
  $('targetImage').alt=packing?'Inventory of fixed rectangles':'Target image';$('bestImage').alt=packing?'Best rectangle packing layout':'Best generated image';
  document.querySelector('.bottom-note').textContent=packing?'Each cell is a complete packing layout.':'Each cell is a complete image program.';
}
function expression(n){
  const a=ref(n.a),b=ref(n.b),p=n.parameter.toFixed(3);
  return [()=>p,()=>`${a} + ${b}`,()=>`${a} − ${b}`,()=>`${a} × ${b}`,()=>`safeDiv(${a}, ${b})`,()=>`sin(${a} × ${p})`,()=>`cos(${a} × ${p})`,()=>`abs(${a})`,()=>`min(${a}, ${b})`,()=>`max(${a}, ${b})`,()=>`tanh(${a})`,()=>`step(${a}, ${b})`][n.op]();
}
function drawProgram(s){
  const fragment=document.createDocumentFragment(),active=new Set(s.active_nodes);
  const nodes=s.program?.nodes??[];
  if(s.layout){
    s.layout.placements.forEach(p=>{const row=document.createElement('div');row.className='instruction';const id=document.createElement('span');id.className='reg';id.textContent=String(p.id+1).padStart(2,'0');const text=document.createElement('span');text.textContent=`${p.width}×${p.height} @ ${p.x},${p.y}`;const rotate=document.createElement('span');rotate.className='channel';rotate.textContent=p.rotated?'↻':'';row.append(id,text,rotate);fragment.append(row);});
  }
  nodes.forEach((n,i)=>{
    const row=document.createElement('div');row.className='instruction'+(active.has(i)?'':' inactive');
    const reg=document.createElement('span');reg.className='reg';reg.textContent=`v${i+6}`;
    const exp=document.createElement('span');exp.textContent=expression(n);
    const channel=document.createElement('span');channel.className='channel';channel.textContent=['R','G','B'][i-nodes.length+3]??'';
    row.append(reg,exp,channel);fragment.append(row);
  });
  $('program').replaceChildren(fragment);
}
function drawPopulation(s){
  const canvas=$('populationMap'),ctx=canvas.getContext('2d'),w=s.config.width,h=s.config.height;
  canvas.width=w*4;canvas.height=h*4;
  canvas.style.width=`${100*w/Math.max(w,h)}%`;
  const lo=Math.min(...s.scores),hi=Math.max(...s.scores),span=Math.max(hi-lo,1e-9);
  const stops=[[45,52,48],[77,119,94],[175,199,138],[237,182,108]];
  s.scores.forEach((score,i)=>{
    const t=Math.min(2.9999,Math.max(0,(1-(score-lo)/span)*3)),a=Math.floor(t),f=t-a;
    const c=stops[a].map((v,k)=>Math.round(v+(stops[a+1][k]-v)*f));
    ctx.fillStyle=`rgb(${c.join(',')})`;ctx.fillRect((i%w)*4,Math.floor(i/w)*4,4,4);
  });
  const x=(s.selected%w)*4,y=Math.floor(s.selected/w)*4;
  ctx.strokeStyle='#fff';ctx.lineWidth=1;ctx.strokeRect(x+.5,y+.5,3,3);
  $('populationMap').setAttribute('aria-label',`${w} by ${h} population. Selected cell ${s.selected%w}, ${Math.floor(s.selected/w)}. Click or use arrow keys to inspect.`);
}
function drawChart(){
  if(!state)return;
  const canvas=$('chart'),rect=canvas.getBoundingClientRect(),dpr=devicePixelRatio||1;
  canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);
  const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);
  const w=rect.width,h=rect.height,l=44,r=18,t=10,b=25,history=state.history;
  if(!history.length)return;
  let min=Infinity,max=-Infinity;
  history.forEach(p=>{min=Math.min(min,Math.log10(Math.max(p.best,1e-8)));max=Math.max(max,Math.log10(Math.max(p.mean,1e-8)));});
  const span=Math.max(.2,max-min);min-=span*.1;max+=span*.1;
  const last=Math.max(1,history.at(-1).generation),x=g=>l+g/last*(w-l-r),y=v=>t+(max-Math.log10(Math.max(v,1e-8)))/(max-min)*(h-t-b);
  ctx.font='8px Consolas,monospace';ctx.fillStyle='#929b8f';ctx.strokeStyle='#343a30';ctx.lineWidth=1;
  for(let i=0;i<4;i++){
    const yy=t+(h-t-b)*i/3,v=10**(max-(max-min)*i/3);
    ctx.beginPath();ctx.moveTo(l,yy+.5);ctx.lineTo(w-r,yy+.5);ctx.stroke();
    ctx.textAlign='right';ctx.fillText(v<.01?v.toExponential(0):v.toFixed(2),l-7,yy+3);
  }
  ctx.textAlign='left';ctx.fillText('0',l,h-7);ctx.textAlign='right';ctx.fillText(integer(state.stats.generation)+' gen',w-r,h-7);
  for(const [key,color,dash] of [['mean','#858f7e',[3,4]],['best','#b4cf94',[]]]){
    ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=1.7;ctx.setLineDash(dash);
    history.forEach((p,i)=>i?ctx.lineTo(x(p.generation),y(p[key])):ctx.moveTo(x(p.generation),y(p[key])));ctx.stroke();
    if(history.length===1){ctx.beginPath();ctx.arc(x(0),y(history[0][key]),2,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();}
  }
  ctx.setLineDash([]);
}
function update(s){
  state=s;
  const packing=s.config.problem==='packing';presentation(packing);
  const key=JSON.stringify([s.config,s.backend_key,s.target_name]);
  if(!initialized||key!==controlsKey){syncControls(s);controlsKey=key;initialized=true;}
  $('device').textContent=s.backend;
  $('generation').textContent=integer(s.stats.generation);
  $('bestScore').textContent=packing?integer(s.best_layout.area):s.stats.best_score.toFixed(5);
  $('speed').replaceChildren(document.createTextNode(s.generations_per_second.toFixed(1)),Object.assign(document.createElement('small'),{textContent:'gen/s'}));
  $('cells').replaceChildren(document.createTextNode(integer(s.config.width*s.config.height)),Object.assign(document.createElement('small'),{textContent:packing?'layouts':'programs'}));
  $('runState').textContent=s.running?'RUNNING':'PAUSED';$('runState').classList.toggle('running',s.running);
  $('run').textContent=s.running?'Ⅱ   Pause evolution':'▶   Run evolution';
  $('step').disabled=s.running;
  $('targetName').textContent=s.target_name.toUpperCase();
  $('gridLabel').textContent=`${s.config.width} × ${s.config.height}`;
  $('selectionLabel').textContent=s.following_best?'Following the best individual':`Inspecting cell (${s.selected%s.config.width}, ${Math.floor(s.selected/s.config.width)})`;
  $('cellBadge').textContent=`CELL ${s.selected%s.config.width}, ${Math.floor(s.selected/s.config.width)}`;
  $('activeNodes').textContent=packing?`${(s.layout.utilization*100).toFixed(1)}% · ${s.layout.width}×${s.layout.height}`:`${s.active_nodes.length} / ${s.config.nodes}`;
  $('follow').disabled=s.following_best;
  $('sampleInfo').textContent=packing?`Filled ${(s.best_layout.utilization*100).toFixed(1)}% · Area lower bound ${integer(s.best_layout.rectangle_area)} · No overlaps`:`${integer(s.config.sample_resolution**2)} training samples · Validation MSE ${s.validation_mse.toFixed(5)} (64²)`;
  const initial=s.history[0]?.best??s.stats.best_score;
  $('improvement').textContent=s.stats.generation?`${Math.max(0,packing?(1-(1-initial)/(1-s.stats.best_score))*100:(initial>0?(1-s.stats.best_score/initial)*100:0)).toFixed(1)}% ${packing?'smaller footprint':'less error'} since generation 0`:'Ready for the first generation';
  $('error').hidden=!s.error;$('error').textContent=s.error??'';
  if(s.revision!==revision){
    $('targetImage').src=`/target.png?v=${s.revision}`;
    $('bestImage').src=`/best.png?v=${s.revision}`;
    $('selectedImage').src=`/selected.png?v=${s.revision}`;
    drawProgram(s);drawPopulation(s);drawChart();revision=s.revision;
  }
}
async function poll(){
  try{const r=await fetch('/api/state');if(!r.ok)throw new Error('Engine unavailable');update(await r.json());}
  catch(e){$('device').textContent='Engine disconnected';$('error').hidden=false;$('error').textContent='The local engine is unavailable. Restart Revo 2 in the terminal to reconnect.';}
  finally{setTimeout(poll,500);}
}
$('run').addEventListener('click',()=>state&&action(state.running?'pause':'start'));
$('step').addEventListener('click',()=>action('step'));
$('follow').addEventListener('click',()=>action('follow'));
$('problem').addEventListener('change',()=>formProblem($('problem').value));
$('mutation').addEventListener('input',()=>$('mutationValue').textContent=$('mutation').value+'%');
$('settings').addEventListener('submit',async e=>{
  e.preventDefault();if(!state)return;
  const config={...state.config,problem:$('problem').value,rectangle_count:Number($('rectangleCount').value),rectangle_seed:Number($('rectangleSeed').value),width:Number($('populationWidth').value),height:Number($('populationHeight').value),nodes:Number($('nodes').value),seed:Number($('seed').value),mutation_rate:Number($('mutation').value)/100};
  $('reset').disabled=true;
  await action('reset',{config,backend:$('backend').value,preset:$('preset').value||null});
  setTimeout(()=>$('reset').disabled=false,900);
});
$('upload').addEventListener('change',async e=>{
  const file=e.target.files[0];if(!file)return;
  if(file.size>10*1024*1024){toast('Please choose an image smaller than 10 MB.');return;}
  try{const r=await fetch('/api/target',{method:'POST',headers:{'Content-Type':'application/octet-stream'},body:file});if(!r.ok)throw new Error(await r.text());toast('Target uploaded. A fresh population is being prepared.');}
  catch(error){toast(error.message);}finally{e.target.value='';}
});
$('populationMap').addEventListener('click',e=>{
  if(!state)return;const r=e.currentTarget.getBoundingClientRect(),w=state.config.width,h=state.config.height;
  const x=Math.min(w-1,Math.floor((e.clientX-r.left)/r.width*w)),y=Math.min(h-1,Math.floor((e.clientY-r.top)/r.height*h));
  action('select',{index:y*w+x});
});
$('populationMap').addEventListener('keydown',e=>{
  if(!state)return;const w=state.config.width,h=state.config.height,i=state.selected,x=i%w,y=Math.floor(i/w);
  const moves={ArrowLeft:y*w+(x+w-1)%w,ArrowRight:y*w+(x+1)%w,ArrowUp:((y+h-1)%h)*w+x,ArrowDown:((y+1)%h)*w+x};
  if(e.key in moves){e.preventDefault();action('select',{index:moves[e.key]});}
});
window.addEventListener('resize',drawChart);
poll();
