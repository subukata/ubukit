from pathlib import Path

def fn(rows,width,name):
 p=['(func $'+name+' (param $xp i32) (param $cp i32) (param $n i32) (param $d i32) (param $k i32) (param $stride i32) (param $lp i32) (param $dp i32) (param $fp i32) (result i32)',
 '(local $i i32) (local $c i32) (local $f i32) (local $x v128) (local $delta v128) (local $a f64)']
 for r in range(rows):p += [f'(local $best{r} i32) (local $unsafe{r} i32) (local $distance{r} f64)']
 for t in range(width//2):p +=[f'(local $center{t} v128)']
 for r in range(rows):
  for t in range(width//2):p +=[f'(local $sum{r}_{t} v128)']
 p+=['(block $rd (loop $rn',f'(br_if $rd (i32.gt_u (i32.add (local.get $i) (i32.const {rows})) (local.get $n)))']
 for r in range(rows):p +=[f'(local.set $distance{r} (f64.const inf)) (local.set $best{r} (i32.const 0)) (local.set $unsafe{r} (i32.const 0))']
 p+=['(local.set $c (i32.const 0)) (block $cd (loop $cn (br_if $cd (i32.ge_u (local.get $c) (local.get $k)))']
 for r in range(rows):
  for t in range(width//2):p +=[f'(local.set $sum{r}_{t} (v128.const f64x2 0 0))']
 p+=['(local.set $f (i32.const 0)) (block $fd (loop $fn (br_if $fd (i32.ge_u (local.get $f) (local.get $d)))']
 for t in range(width//2):p +=[f'(local.set $center{t} (v128.load (i32.add (local.get $cp) (i32.mul (i32.add (i32.mul (local.get $f) (local.get $stride)) (i32.add (local.get $c) (i32.const {2*t}))) (i32.const 8)))))']
 for r in range(rows):
  p +=[f'(local.set $x (f64x2.splat (f64.load (i32.add (local.get $xp) (i32.mul (i32.add (i32.mul (i32.add (local.get $i) (i32.const {r})) (local.get $d)) (local.get $f)) (i32.const 8))))))']
  for t in range(width//2):p +=[f'(local.set $delta (f64x2.sub (local.get $x) (local.get $center{t}))) (local.set $sum{r}_{t} (f64x2.add (local.get $sum{r}_{t}) (f64x2.mul (local.get $delta) (local.get $delta))))']
 p+=['(local.set $f (i32.add (local.get $f) (i32.const 1))) (br $fn)))']
 for t in range(width//2):
  for lane in range(2):
   offset=2*t+lane;p +=[f'(if (i32.lt_u (i32.add (local.get $c) (i32.const {offset})) (local.get $k)) (then']
   for r in range(rows):p +=[f'''(local.set $a (f64x2.extract_lane {lane} (local.get $sum{r}_{t})))
    (if (i32.or (f64.ne (local.get $a) (local.get $a)) (i32.or (f64.gt (local.get $a) (f64.const 1.7976931348623157e308)) (i32.and (f64.gt (local.get $a) (f64.const 0)) (f64.lt (local.get $a) (f64.const 2.2250738585072014e-308))))) (then (local.set $unsafe{r} (i32.const 1))))
    (if (f64.eq (local.get $a) (f64.const 0)) (then
     (local.set $f (i32.const 0))
     (block $zero_done (loop $zero_next
      (br_if $zero_done (i32.ge_u (local.get $f) (local.get $d)))
      (if (f64.ne
       (f64.load (i32.add (local.get $xp) (i32.mul (i32.add (i32.mul (i32.add (local.get $i) (i32.const {r})) (local.get $d)) (local.get $f)) (i32.const 8))))
       (f64.load (i32.add (local.get $cp) (i32.mul (i32.add (i32.mul (local.get $f) (local.get $stride)) (i32.add (local.get $c) (i32.const {offset}))) (i32.const 8)))))
       (then (local.set $unsafe{r} (i32.const 1)) (br $zero_done)))
      (local.set $f (i32.add (local.get $f) (i32.const 1))) (br $zero_next)
     ))
    ))
    (if (f64.lt (local.get $a) (local.get $distance{r})) (then (local.set $distance{r} (local.get $a)) (local.set $best{r} (i32.add (local.get $c) (i32.const {offset})))))''']
   p+=['))']
 p +=[f'(local.set $c (i32.add (local.get $c) (i32.const {width}))) (br $cn)))']
 for r in range(rows):
  for ptr,ctype,size,value in [('lp','i32',4,f'best{r}'),('dp','f64',8,f'distance{r}'),('fp','i32',4,f'unsafe{r}')]:p +=[f'({ctype}.store (i32.add (local.get ${ptr}) (i32.mul (i32.add (local.get $i) (i32.const {r})) (i32.const {size}))) (local.get ${value}))']
 p +=[f'(local.set $i (i32.add (local.get $i) (i32.const {rows}))) (br $rn))) (local.get $i))']
 return '\n'.join(p)

for rows,width in [(1,4),(1,8),(2,4),(2,8),(4,4),(4,8)]:
 name=f'r{rows}w{width}';text='(module (memory (export "memory") 1)\n'+fn(rows,width,'main')+'\n'+fn(1,width,'tail')
 text+='''\n(func (export "nearest") (param $xp i32) (param $cp i32) (param $n i32) (param $d i32) (param $k i32) (param $stride i32) (param $lp i32) (param $dp i32) (param $fp i32) (local $done i32)
 (local.set $done (call $main (local.get $xp) (local.get $cp) (local.get $n) (local.get $d) (local.get $k) (local.get $stride) (local.get $lp) (local.get $dp) (local.get $fp)))
 (drop (call $tail (i32.add (local.get $xp) (i32.mul (i32.mul (local.get $done) (local.get $d)) (i32.const 8))) (local.get $cp) (i32.sub (local.get $n) (local.get $done)) (local.get $d) (local.get $k) (local.get $stride) (i32.add (local.get $lp) (i32.mul (local.get $done) (i32.const 4))) (i32.add (local.get $dp) (i32.mul (local.get $done) (i32.const 8))) (i32.add (local.get $fp) (i32.mul (local.get $done) (i32.const 4))))))\n)'''
 Path(__file__).with_name('kmeans_'+name+'.wat').write_text(text)
