from pathlib import Path
p=Path(__file__).with_name('radius_candidates.wat');s=p.read_text();assert '(export "scan_f32_candidates")' not in s
out=[''' (func (export "scan_f32_candidates")
  (param $base i32) (param $i i32) (param $n i32) (param $d i32)
  (param $start i32) (param $end i32) (param $cutoff f32) (param $out i32) (result i32)
  (local $j i32) (local $f i32) (local $off i32) (local $count i32) (local $mask i32)
  (local $s0 v128) (local $s1 v128) (local $s2 v128) (local $s3 v128) (local $query v128) (local $diff v128)
  (local $sq f32) (local $tail f32)
  (local.set $j (local.get $start))''']
for size in [16,4]:
 lanes=size//4;tag=f'p{size}'
 out += [f'  (block $end_{tag} (loop $loop_{tag}',f'   (br_if $end_{tag} (i32.ge_u (i32.add (local.get $j) (i32.const {size-1})) (local.get $end)))','   (local.set $f (i32.const 0))']
 for v in range(lanes):out.append(f'   (local.set $s{v} (v128.const f32x4 0 0 0 0))')
 out += [f'   (block $endf_{tag} (loop $loopf_{tag}',f'    (br_if $endf_{tag} (i32.ge_u (local.get $f) (local.get $d)))','    (local.set $off (i32.add (local.get $base) (i32.mul (i32.mul (local.get $f) (local.get $n)) (i32.const 4))))','    (local.set $query (f32x4.splat (f32.load (i32.add (local.get $off) (i32.mul (local.get $i) (i32.const 4))))))']
 for v in range(lanes):out += [f'    (local.set $diff (f32x4.sub (local.get $query) (v128.load offset={16*v} align=4 (i32.add (local.get $off) (i32.mul (local.get $j) (i32.const 4))))))',f'    (local.set $s{v} (f32x4.add (local.get $s{v}) (f32x4.mul (local.get $diff) (local.get $diff))))']
 mask='(f32x4.gt (local.get $s0) (f32x4.splat (local.get $cutoff)))'
 for v in range(1,lanes):mask=f'(v128.and {mask} (f32x4.gt (local.get $s{v}) (f32x4.splat (local.get $cutoff))))'
 out += ['    (local.set $f (i32.add (local.get $f) (i32.const 1)))',f'    (if (i32.eqz (i32.and (local.get $f) (i32.const 7))) (then (br_if $endf_{tag} (i8x16.all_true {mask}))))',f'    (br $loopf_{tag})))']
 for v in range(lanes):
  out.append(f'   (local.set $mask (i32x4.bitmask (f32x4.le (local.get $s{v}) (f32x4.splat (local.get $cutoff)))))')
  for lane in range(4):
   out += [f'   (if (i32.and (local.get $mask) (i32.const {1<<lane})) (then',f'    (i32.store (i32.add (local.get $out) (i32.mul (local.get $count) (i32.const 4))) (i32.add (local.get $j) (i32.const {v*4+lane})))','    (local.set $count (i32.add (local.get $count) (i32.const 1)))))']
 out += [f'   (local.set $j (i32.add (local.get $j) (i32.const {size})))',f'   (br $loop_{tag})))']
out += ['''  (block $end_tail (loop $loop_tail
   (br_if $end_tail (i32.ge_u (local.get $j) (local.get $end)))
   (local.set $f (i32.const 0)) (local.set $sq (f32.const 0))
   (block $endf_tail (loop $loopf_tail
    (br_if $endf_tail (i32.ge_u (local.get $f) (local.get $d)))
    (local.set $off (i32.add (local.get $base) (i32.mul (i32.mul (local.get $f) (local.get $n)) (i32.const 4))))
    (local.set $tail (f32.sub (f32.load (i32.add (local.get $off) (i32.mul (local.get $i) (i32.const 4)))) (f32.load (i32.add (local.get $off) (i32.mul (local.get $j) (i32.const 4))))))
    (local.set $sq (f32.add (local.get $sq) (f32.mul (local.get $tail) (local.get $tail))))
    (br_if $endf_tail (f32.gt (local.get $sq) (local.get $cutoff)))
    (local.set $f (i32.add (local.get $f) (i32.const 1))) (br $loopf_tail)))
   (if (f32.le (local.get $sq) (local.get $cutoff)) (then
    (i32.store (i32.add (local.get $out) (i32.mul (local.get $count) (i32.const 4))) (local.get $j))
    (local.set $count (i32.add (local.get $count) (i32.const 1)))))
   (local.set $j (i32.add (local.get $j) (i32.const 1))) (br $loop_tail)))
  (local.get $count))''']
fragment='\n'.join(out);Path(__file__).with_name('f32_candidates.wat').write_text(fragment+'\n');s=s.rstrip();assert s.endswith(')');p.write_text(s[:-1]+fragment+'\n)\n')
