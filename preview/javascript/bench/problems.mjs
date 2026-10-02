/** Synthetic objectives, intentionally spanning favorable and difficult settings. */
const range = (low, high, log = false) => ({type: 'float', low, high, log});
const axes = (n, low, high) => Object.fromEntries(Array.from({length:n}, (_,i) => [`x${i}`,range(low,high)]));
export const problems = [
  {name:'sphere_4d', space:axes(4,-5,5), objective:p=>Object.values(p).reduce((s,x)=>s+x*x,0), optimum:0},
  {name:'anisotropic_6d', space:axes(6,-5,5), objective:p=>Object.values(p).reduce((s,x,i)=>s+10**(i/2)*(x-.3)**2,0), optimum:0},
  {name:'rosenbrock_3d', space:axes(3,-2,2), objective:p=>100*(p.x1-p.x0*p.x0)**2+(1-p.x0)**2+100*(p.x2-p.x1*p.x1)**2+(1-p.x1)**2, optimum:0},
  {name:'rotated_2d', space:axes(2,-5,5), objective:p=>(p.x0+p.x1-.6)**2+100*(p.x0-p.x1)**2, optimum:0},
  {name:'rastrigin_4d', space:axes(4,-5.12,5.12), objective:p=>40+Object.values(p).reduce((s,x)=>s+x*x-10*Math.cos(2*Math.PI*x),0), optimum:0},
  {name:'mixed', space:{rate:range(1e-5,1,true),depth:{type:'int',low:1,high:12},mode:{type:'categorical',choices:['a','b','c','d']}}, objective:p=>(Math.log10(p.rate)+2.5)**2+(p.depth-7)**2/8+({a:2,b:0,c:1,d:3}[p.mode]), optimum:0},
  {name:'log_scale_3d',space:{a:range(1e-12,1e3,true),b:range(1e-8,1e8,true),c:range(1e-6,1e2,true)},objective:p=>(Math.log10(p.a)+4)**2+(Math.log10(p.b)-1)**2+(Math.log10(p.c)+1)**2,optimum:0},
  {name:'boundary_4d',space:axes(4,0,1),objective:p=>Object.values(p).reduce((s,x)=>s+x*x,0),optimum:0},
  {name:'categorical_interaction',space:{a:{type:'categorical',choices:[0,1,2,3,4]},b:{type:'categorical',choices:[0,1,2,3,4]},x:range(-1,1)},objective:p=>(p.a===p.b?0:3)+(p.a===3?0:1)+(p.x-.25)**2,optimum:0}
];
