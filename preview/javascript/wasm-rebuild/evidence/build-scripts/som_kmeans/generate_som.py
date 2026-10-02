from pathlib import Path

def fn(rows,width,name,grid):
 p=['(func $'+name+' (param $xp i32) (param $cp i32) (param $n i32) (param $d i32) (param $m i32) (param $stride i32) (param $out i32) (param $fp i32) (param $gamma f64) (result i32)',
 '(local $i i32) (local $j i32) (local $f i32) (local $x v128) (local $delta v128) (local $a f64) (local $address i32) (local $code i32)']
 for r in range(rows):p += [f'(local $unsafe{r} i32)']
 for t in range(width//2):p +=[f'(local $center{t} v128)']
 for r in range(rows):
  for t in range(width//2):p +=[f'(local $sum{r}_{t} v128)']
 p+=['(block $rd (loop $rn',f'(br_if $rd (i32.gt_u (i32.add (local.get $i) (i32.const {rows})) (local.get $n)))']
 for r in range(rows):p +=[f'(local.set $unsafe{r} '+(f'(i32.load (i32.add (local.get $fp) (i32.mul (i32.add (local.get $i) (i32.const {r})) (i32.const 4)))))'if grid else'(i32.const 0))')]
 p+=['(local.set $j (i32.const 0)) (block $cd (loop $cn (br_if $cd (i32.ge_u (local.get $j) (local.get $m)))']
 for r in range(rows):
  for t in range(width//2):p +=[f'(local.set $sum{r}_{t} (v128.const f64x2 0 0))']
 p+=['(local.set $f (i32.const 0)) (block $fd (loop $fn (br_if $fd (i32.ge_u (local.get $f) (local.get $d)))']
 for t in range(width//2):p +=[f'(local.set $center{t} (v128.load (i32.add (local.get $cp) (i32.mul (i32.add (i32.mul (local.get $f) (local.get $stride)) (i32.add (local.get $j) (i32.const {2*t}))) (i32.const 8)))))']
 for r in range(rows):
  p +=[f'(local.set $x (f64x2.splat (f64.load (i32.add (local.get $xp) (i32.mul (i32.add (i32.mul (i32.add (local.get $i) (i32.const {r})) (local.get $d)) (local.get $f)) (i32.const 8))))))']
  for t in range(width//2):p +=[f'(local.set $delta (f64x2.sub (local.get $x) (local.get $center{t}))) (local.set $sum{r}_{t} (f64x2.add (local.get $sum{r}_{t}) (f64x2.mul (local.get $delta) (local.get $delta))))']
 p+=['(local.set $f (i32.add (local.get $f) (i32.const 1))) (br $fn)))']
 def guard(r,offset,kind):
  return f'''(if (i32.eqz (f64.le (local.get $a) (f64.const 1.7976931348623157e308))) (then
   (local.set $code (i32.add (i32.mul (i32.add (local.get $j) (i32.const {offset})) (i32.const 2)) (i32.const {kind})))
   (if (i32.or (i32.eqz (local.get $unsafe{r})) (i32.lt_u (local.get $code) (local.get $unsafe{r}))) (then (local.set $unsafe{r} (local.get $code))))))'''
 for t in range(width//2):
  for lane in range(2):
   offset=2*t+lane;p +=[f'(if (i32.lt_u (i32.add (local.get $j) (i32.const {offset})) (local.get $m)) (then']
   for r in range(rows):
    p +=[f'(local.set $a (f64x2.extract_lane {lane} (local.get $sum{r}_{t})))',guard(r,offset,1),f'(local.set $address (i32.add (local.get $out) (i32.mul (i32.add (i32.mul (i32.add (local.get $i) (i32.const {r})) (local.get $m)) (i32.add (local.get $j) (i32.const {offset}))) (i32.const 8))))']
    if grid:p +=['(local.set $a (f64.add (f64.load (local.get $address)) (f64.mul (local.get $gamma) (local.get $a))))',guard(r,offset,2)]
    p +=['(f64.store (local.get $address) (local.get $a))']
   p+=['))']
 p +=[f'(local.set $j (i32.add (local.get $j) (i32.const {width}))) (br $cn)))']
 for r in range(rows):p +=[f'(i32.store (i32.add (local.get $fp) (i32.mul (i32.add (local.get $i) (i32.const {r})) (i32.const 4))) (local.get $unsafe{r}))']
 p +=[f'(local.set $i (i32.add (local.get $i) (i32.const {rows}))) (br $rn))) (local.get $i))']
 return '\n'.join(p)

text='(module (memory (export "memory") 1)\n'+Path(__file__).with_name('som_prototype.wat').read_text().replace('(func (export "prototype")','(func $prototype_solo')+'\n'+Path(__file__).with_name('som_prototype4.wat').read_text()
for grid in [False,True]:
 name='grid' if grid else'feature';text+='\n'+fn(2,8,name+'main',grid)+'\n'+fn(1,8,name+'tail',grid)
 text+=f'''\n(func (export "{name}") (param $xp i32) (param $cp i32) (param $n i32) (param $d i32) (param $m i32) (param $stride i32) (param $out i32) (param $fp i32) (param $gamma f64) (local $done i32)
 (local.set $done (call ${name}main (local.get $xp) (local.get $cp) (local.get $n) (local.get $d) (local.get $m) (local.get $stride) (local.get $out) (local.get $fp) (local.get $gamma)))
 (drop (call ${name}tail (i32.add (local.get $xp) (i32.mul (i32.mul (local.get $done) (local.get $d)) (i32.const 8))) (local.get $cp) (i32.sub (local.get $n) (local.get $done)) (local.get $d) (local.get $m) (local.get $stride) (i32.add (local.get $out) (i32.mul (i32.mul (local.get $done) (local.get $m)) (i32.const 8))) (i32.add (local.get $fp) (i32.mul (local.get $done) (i32.const 4))) (local.get $gamma))))\n'''
Path(__file__).with_name('som_train.wat').write_text(text+')\n')
