/** Comparison baseline, not production runtime: independently written one-pass
 * Map contingency, Number ARI, and ungrouped scalar hypergeometric expectation.
 * Uses the same stable mathematical definition; no pair enumeration or Monte Carlo.
 * The recurrence repeats work for every cluster pair, without size compression.
 */
function table(a,b){
  const n=a.length,ai=new Map(),bi=new Map(),rows=[],cols=[],cells=new Map();
  let ak,bk;
  for(let i=0;i<n;i++){
    const x=a[i],y=b[i],xt=typeof x,yt=typeof y;
    if((xt!=='string' && !Number.isSafeInteger(x)) || (yt!=='string' && !Number.isSafeInteger(y)) || (ak!==undefined&&ak!==xt) || (bk!==undefined&&bk!==yt))throw new TypeError('invalid label');
    ak=xt;bk=yt;
    let r=ai.get(x),c=bi.get(y);
    if(r===undefined){r=rows.length;ai.set(x,r);rows.push(0);}
    if(c===undefined){c=cols.length;bi.set(y,c);cols.push(0);}
    rows[r]++;cols[c]++;
    const key=r*(n+1)+c;cells.set(key,(cells.get(key)||0)+1);
  }
  return{n,rows,cols,cells};
}
function rand({n,rows,cols,cells}){
  const c2=x=>x*(x-1)/2;
  let intersection=0,r=0,c=0;for(const x of cells.values())intersection+=c2(x);for(const x of rows)r+=c2(x);for(const x of cols)c+=c2(x);
  if(intersection===r && intersection===c)return 1;
  const expected=r*c/c2(n);return (intersection-expected)/((r+c)/2-expected);
}
const entropy=(counts,n)=>counts.reduce((v,c)=>v-(c/n)*Math.log(c/n),0);
function expectation(n,row,col){
  if(col===1)return 0;
  const lower=Math.max(0,row+col-n),upper=Math.min(row,col);
  const mode=Math.max(lower,Math.min(upper,Math.floor((row+1)*(col+1)/(n+2))));
  const contribution=x=>x===0?0:x/n*Math.log1p((col-x)/x);
  let mass=1,value=contribution(mode),p=1;
  for(let x=mode+1;x<=upper;x++){p*=((row-x+1)/x)*((col-x+1)/(n-row-col+x));mass+=p;if(p!==0)value+=p*contribution(x);}
  p=1;for(let x=mode-1;x>=lower;x--){p*=((x+1)/(row-x))*((n-row-col+x+1)/(col-x));mass+=p;if(p!==0)value+=p*contribution(x);}
  return value/mass;
}
function mutual(t){
  const{n,rows,cols,cells}=t,ka=rows.length,kb=cols.length;
  if(ka===kb&&(ka<2||cells.size===ka))return 1;if(ka===1||kb===1||ka===n||kb===n)return 0;
  const ha=entropy(rows,n),hb=entropy(cols,n),forward=ha<=hb,first=forward?rows:cols,other=forward?cols:rows;
  let expected=0,observed=0;
  for(const a of first)for(const b of other)expected+=expectation(n,a,b);
  for(const[key,x]of cells){const r=Math.floor(key/(n+1)),c=key%(n+1);observed+=x/n*Math.log1p(((forward?cols[c]:rows[r])-x)/x);}
  return(expected-observed)/(expected+Math.abs(ha-hb)/2);
}
export const baselineARI=(a,b)=>rand(table(a,b));
export const baselineAMI=(a,b)=>mutual(table(a,b));
export function baselineJoint(a,b){const t=table(a,b);return{ari:rand(t),ami:mutual(t)};}
