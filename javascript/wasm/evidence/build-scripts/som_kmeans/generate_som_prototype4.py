from pathlib import Path
p=['''(func (export "prototype") (param $xp i32) (param $pp i32) (param $rp i32) (param $np i32) (param $dp i32) (param $vp i32) (param $n i32) (param $d i32) (param $m i32) (param $q i32)
(local $i i32) (local $j i32) (local $f i32) (local $h i32) (local $xo i32) (local $vo i32) (local $address i32) (local $x v128) (local $value v128) (local $xs f64) (local $vs f64)''']
for t in range(4):p +=[f'(local $p{t} f64) (local $pv{t} v128) (local $no{t} i32) (local $ro{t} i32)']
p +=['''(block $id (loop $in (br_if $id (i32.ge_u (local.get $i) (local.get $n)))
(local.set $xo (i32.add (local.get $xp) (i32.mul (i32.mul (local.get $i) (local.get $d)) (i32.const 8))))
(local.set $vo (i32.add (local.get $vp) (i32.mul (i32.mul (local.get $i) (local.get $q)) (i32.const 8))))
(local.set $j (i32.const 0))
(block $jd (loop $jn (br_if $jd (i32.ge_u (i32.add (local.get $j) (i32.const 3)) (local.get $m)))''']
for t in range(4):p +=[f'''(local.set $p{t} (f64.load (i32.add (local.get $pp) (i32.mul (i32.add (i32.mul (local.get $i) (local.get $m)) (i32.add (local.get $j) (i32.const {t}))) (i32.const 8)))))
(local.set $pv{t} (f64x2.splat (local.get $p{t})))
(local.set $address (i32.add (local.get $dp) (i32.mul (i32.add (local.get $j) (i32.const {t})) (i32.const 8))))
(f64.store (local.get $address) (f64.add (f64.load (local.get $address)) (local.get $p{t})))
(local.set $no{t} (i32.add (local.get $np) (i32.mul (i32.mul (i32.add (local.get $j) (i32.const {t})) (local.get $d)) (i32.const 8))))
(local.set $ro{t} (i32.add (local.get $rp) (i32.mul (i32.mul (i32.add (local.get $j) (i32.const {t})) (local.get $q)) (i32.const 8))))''']
p +=['''(local.set $f (i32.const 0))
(block $fd (loop $fn (br_if $fd (i32.ge_u (i32.add (local.get $f) (i32.const 1)) (local.get $d)))
(local.set $x (v128.load (i32.add (local.get $xo) (i32.mul (local.get $f) (i32.const 8)))))''']
for t in range(4):p +=[f'''(local.set $address (i32.add (local.get $no{t}) (i32.mul (local.get $f) (i32.const 8))))
(v128.store (local.get $address) (f64x2.add (v128.load (local.get $address)) (f64x2.mul (local.get $pv{t}) (local.get $x))))''']
p +=['''(local.set $f (i32.add (local.get $f) (i32.const 2))) (br $fn)))
(if (i32.lt_u (local.get $f) (local.get $d)) (then
(local.set $xs (f64.load (i32.add (local.get $xo) (i32.mul (local.get $f) (i32.const 8)))))''']
for t in range(4):p +=[f'''(local.set $address (i32.add (local.get $no{t}) (i32.mul (local.get $f) (i32.const 8))))
(f64.store (local.get $address) (f64.add (f64.load (local.get $address)) (f64.mul (local.get $p{t}) (local.get $xs))))''']
p +=['''))
(local.set $h (i32.const 0))
(block $hd (loop $hn (br_if $hd (i32.ge_u (i32.add (local.get $h) (i32.const 1)) (local.get $q)))
(local.set $address (i32.add (local.get $vo) (i32.mul (local.get $h) (i32.const 8))))
(local.set $value (v128.load (local.get $address)))''']
for t in range(4):p +=[f'(local.set $value (f64x2.add (local.get $value) (f64x2.mul (local.get $pv{t}) (v128.load (i32.add (local.get $ro{t}) (i32.mul (local.get $h) (i32.const 8)))))))']
p +=['''(v128.store (local.get $address) (local.get $value))
(local.set $h (i32.add (local.get $h) (i32.const 2))) (br $hn)))
(if (i32.lt_u (local.get $h) (local.get $q)) (then
(local.set $address (i32.add (local.get $vo) (i32.mul (local.get $h) (i32.const 8))))
(local.set $vs (f64.load (local.get $address)))''']
for t in range(4):p +=[f'(local.set $vs (f64.add (local.get $vs) (f64.mul (local.get $p{t}) (f64.load (i32.add (local.get $ro{t}) (i32.mul (local.get $h) (i32.const 8)))))))']
p +=['''(f64.store (local.get $address) (local.get $vs))))
(local.set $j (i32.add (local.get $j) (i32.const 4))) (br $jn)))
(if (i32.lt_u (local.get $j) (local.get $m)) (then
(call $prototype_solo (local.get $xo)
(i32.add (local.get $pp) (i32.mul (i32.add (i32.mul (local.get $i) (local.get $m)) (local.get $j)) (i32.const 8)))
(i32.add (local.get $rp) (i32.mul (i32.mul (local.get $j) (local.get $q)) (i32.const 8)))
(i32.add (local.get $np) (i32.mul (i32.mul (local.get $j) (local.get $d)) (i32.const 8)))
(i32.add (local.get $dp) (i32.mul (local.get $j) (i32.const 8)))
(local.get $vo) (i32.const 1) (local.get $d) (i32.sub (local.get $m) (local.get $j)) (local.get $q))))
(local.set $i (i32.add (local.get $i) (i32.const 1))) (br $in)))
)''']
Path(__file__).with_name('som_prototype4.wat').write_text('\n'.join(p)+'\n')
