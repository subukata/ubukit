 (func (export "scan_f32_candidates")
  (param $base i32) (param $i i32) (param $n i32) (param $d i32)
  (param $start i32) (param $end i32) (param $cutoff f32) (param $out i32) (result i32)
  (local $j i32) (local $f i32) (local $off i32) (local $count i32) (local $mask i32)
  (local $s0 v128) (local $s1 v128) (local $s2 v128) (local $s3 v128) (local $query v128) (local $diff v128)
  (local $sq f32) (local $tail f32)
  (local.set $j (local.get $start))
  (block $end_p16 (loop $loop_p16
   (br_if $end_p16 (i32.ge_u (i32.add (local.get $j) (i32.const 15)) (local.get $end)))
   (local.set $f (i32.const 0))
   (local.set $s0 (v128.const f32x4 0 0 0 0))
   (local.set $s1 (v128.const f32x4 0 0 0 0))
   (local.set $s2 (v128.const f32x4 0 0 0 0))
   (local.set $s3 (v128.const f32x4 0 0 0 0))
   (block $endf_p16 (loop $loopf_p16
    (br_if $endf_p16 (i32.ge_u (local.get $f) (local.get $d)))
    (local.set $off (i32.add (local.get $base) (i32.mul (i32.mul (local.get $f) (local.get $n)) (i32.const 4))))
    (local.set $query (f32x4.splat (f32.load (i32.add (local.get $off) (i32.mul (local.get $i) (i32.const 4))))))
    (local.set $diff (f32x4.sub (local.get $query) (v128.load offset=0 align=4 (i32.add (local.get $off) (i32.mul (local.get $j) (i32.const 4))))))
    (local.set $s0 (f32x4.add (local.get $s0) (f32x4.mul (local.get $diff) (local.get $diff))))
    (local.set $diff (f32x4.sub (local.get $query) (v128.load offset=16 align=4 (i32.add (local.get $off) (i32.mul (local.get $j) (i32.const 4))))))
    (local.set $s1 (f32x4.add (local.get $s1) (f32x4.mul (local.get $diff) (local.get $diff))))
    (local.set $diff (f32x4.sub (local.get $query) (v128.load offset=32 align=4 (i32.add (local.get $off) (i32.mul (local.get $j) (i32.const 4))))))
    (local.set $s2 (f32x4.add (local.get $s2) (f32x4.mul (local.get $diff) (local.get $diff))))
    (local.set $diff (f32x4.sub (local.get $query) (v128.load offset=48 align=4 (i32.add (local.get $off) (i32.mul (local.get $j) (i32.const 4))))))
    (local.set $s3 (f32x4.add (local.get $s3) (f32x4.mul (local.get $diff) (local.get $diff))))
    (local.set $f (i32.add (local.get $f) (i32.const 1)))
    (if (i32.eqz (i32.and (local.get $f) (i32.const 7))) (then (br_if $endf_p16 (i8x16.all_true (v128.and (v128.and (v128.and (f32x4.gt (local.get $s0) (f32x4.splat (local.get $cutoff))) (f32x4.gt (local.get $s1) (f32x4.splat (local.get $cutoff)))) (f32x4.gt (local.get $s2) (f32x4.splat (local.get $cutoff)))) (f32x4.gt (local.get $s3) (f32x4.splat (local.get $cutoff))))))))
    (br $loopf_p16)))
   (local.set $mask (i32x4.bitmask (f32x4.le (local.get $s0) (f32x4.splat (local.get $cutoff)))))
   (if (i32.and (local.get $mask) (i32.const 1)) (then
    (i32.store (i32.add (local.get $out) (i32.mul (local.get $count) (i32.const 4))) (i32.add (local.get $j) (i32.const 0)))
    (local.set $count (i32.add (local.get $count) (i32.const 1)))))
   (if (i32.and (local.get $mask) (i32.const 2)) (then
    (i32.store (i32.add (local.get $out) (i32.mul (local.get $count) (i32.const 4))) (i32.add (local.get $j) (i32.const 1)))
    (local.set $count (i32.add (local.get $count) (i32.const 1)))))
   (if (i32.and (local.get $mask) (i32.const 4)) (then
    (i32.store (i32.add (local.get $out) (i32.mul (local.get $count) (i32.const 4))) (i32.add (local.get $j) (i32.const 2)))
    (local.set $count (i32.add (local.get $count) (i32.const 1)))))
   (if (i32.and (local.get $mask) (i32.const 8)) (then
    (i32.store (i32.add (local.get $out) (i32.mul (local.get $count) (i32.const 4))) (i32.add (local.get $j) (i32.const 3)))
    (local.set $count (i32.add (local.get $count) (i32.const 1)))))
   (local.set $mask (i32x4.bitmask (f32x4.le (local.get $s1) (f32x4.splat (local.get $cutoff)))))
   (if (i32.and (local.get $mask) (i32.const 1)) (then
    (i32.store (i32.add (local.get $out) (i32.mul (local.get $count) (i32.const 4))) (i32.add (local.get $j) (i32.const 4)))
    (local.set $count (i32.add (local.get $count) (i32.const 1)))))
   (if (i32.and (local.get $mask) (i32.const 2)) (then
    (i32.store (i32.add (local.get $out) (i32.mul (local.get $count) (i32.const 4))) (i32.add (local.get $j) (i32.const 5)))
    (local.set $count (i32.add (local.get $count) (i32.const 1)))))
   (if (i32.and (local.get $mask) (i32.const 4)) (then
    (i32.store (i32.add (local.get $out) (i32.mul (local.get $count) (i32.const 4))) (i32.add (local.get $j) (i32.const 6)))
    (local.set $count (i32.add (local.get $count) (i32.const 1)))))
   (if (i32.and (local.get $mask) (i32.const 8)) (then
    (i32.store (i32.add (local.get $out) (i32.mul (local.get $count) (i32.const 4))) (i32.add (local.get $j) (i32.const 7)))
    (local.set $count (i32.add (local.get $count) (i32.const 1)))))
   (local.set $mask (i32x4.bitmask (f32x4.le (local.get $s2) (f32x4.splat (local.get $cutoff)))))
   (if (i32.and (local.get $mask) (i32.const 1)) (then
    (i32.store (i32.add (local.get $out) (i32.mul (local.get $count) (i32.const 4))) (i32.add (local.get $j) (i32.const 8)))
    (local.set $count (i32.add (local.get $count) (i32.const 1)))))
   (if (i32.and (local.get $mask) (i32.const 2)) (then
    (i32.store (i32.add (local.get $out) (i32.mul (local.get $count) (i32.const 4))) (i32.add (local.get $j) (i32.const 9)))
    (local.set $count (i32.add (local.get $count) (i32.const 1)))))
   (if (i32.and (local.get $mask) (i32.const 4)) (then
    (i32.store (i32.add (local.get $out) (i32.mul (local.get $count) (i32.const 4))) (i32.add (local.get $j) (i32.const 10)))
    (local.set $count (i32.add (local.get $count) (i32.const 1)))))
   (if (i32.and (local.get $mask) (i32.const 8)) (then
    (i32.store (i32.add (local.get $out) (i32.mul (local.get $count) (i32.const 4))) (i32.add (local.get $j) (i32.const 11)))
    (local.set $count (i32.add (local.get $count) (i32.const 1)))))
   (local.set $mask (i32x4.bitmask (f32x4.le (local.get $s3) (f32x4.splat (local.get $cutoff)))))
   (if (i32.and (local.get $mask) (i32.const 1)) (then
    (i32.store (i32.add (local.get $out) (i32.mul (local.get $count) (i32.const 4))) (i32.add (local.get $j) (i32.const 12)))
    (local.set $count (i32.add (local.get $count) (i32.const 1)))))
   (if (i32.and (local.get $mask) (i32.const 2)) (then
    (i32.store (i32.add (local.get $out) (i32.mul (local.get $count) (i32.const 4))) (i32.add (local.get $j) (i32.const 13)))
    (local.set $count (i32.add (local.get $count) (i32.const 1)))))
   (if (i32.and (local.get $mask) (i32.const 4)) (then
    (i32.store (i32.add (local.get $out) (i32.mul (local.get $count) (i32.const 4))) (i32.add (local.get $j) (i32.const 14)))
    (local.set $count (i32.add (local.get $count) (i32.const 1)))))
   (if (i32.and (local.get $mask) (i32.const 8)) (then
    (i32.store (i32.add (local.get $out) (i32.mul (local.get $count) (i32.const 4))) (i32.add (local.get $j) (i32.const 15)))
    (local.set $count (i32.add (local.get $count) (i32.const 1)))))
   (local.set $j (i32.add (local.get $j) (i32.const 16)))
   (br $loop_p16)))
  (block $end_p4 (loop $loop_p4
   (br_if $end_p4 (i32.ge_u (i32.add (local.get $j) (i32.const 3)) (local.get $end)))
   (local.set $f (i32.const 0))
   (local.set $s0 (v128.const f32x4 0 0 0 0))
   (block $endf_p4 (loop $loopf_p4
    (br_if $endf_p4 (i32.ge_u (local.get $f) (local.get $d)))
    (local.set $off (i32.add (local.get $base) (i32.mul (i32.mul (local.get $f) (local.get $n)) (i32.const 4))))
    (local.set $query (f32x4.splat (f32.load (i32.add (local.get $off) (i32.mul (local.get $i) (i32.const 4))))))
    (local.set $diff (f32x4.sub (local.get $query) (v128.load offset=0 align=4 (i32.add (local.get $off) (i32.mul (local.get $j) (i32.const 4))))))
    (local.set $s0 (f32x4.add (local.get $s0) (f32x4.mul (local.get $diff) (local.get $diff))))
    (local.set $f (i32.add (local.get $f) (i32.const 1)))
    (if (i32.eqz (i32.and (local.get $f) (i32.const 7))) (then (br_if $endf_p4 (i8x16.all_true (f32x4.gt (local.get $s0) (f32x4.splat (local.get $cutoff)))))))
    (br $loopf_p4)))
   (local.set $mask (i32x4.bitmask (f32x4.le (local.get $s0) (f32x4.splat (local.get $cutoff)))))
   (if (i32.and (local.get $mask) (i32.const 1)) (then
    (i32.store (i32.add (local.get $out) (i32.mul (local.get $count) (i32.const 4))) (i32.add (local.get $j) (i32.const 0)))
    (local.set $count (i32.add (local.get $count) (i32.const 1)))))
   (if (i32.and (local.get $mask) (i32.const 2)) (then
    (i32.store (i32.add (local.get $out) (i32.mul (local.get $count) (i32.const 4))) (i32.add (local.get $j) (i32.const 1)))
    (local.set $count (i32.add (local.get $count) (i32.const 1)))))
   (if (i32.and (local.get $mask) (i32.const 4)) (then
    (i32.store (i32.add (local.get $out) (i32.mul (local.get $count) (i32.const 4))) (i32.add (local.get $j) (i32.const 2)))
    (local.set $count (i32.add (local.get $count) (i32.const 1)))))
   (if (i32.and (local.get $mask) (i32.const 8)) (then
    (i32.store (i32.add (local.get $out) (i32.mul (local.get $count) (i32.const 4))) (i32.add (local.get $j) (i32.const 3)))
    (local.set $count (i32.add (local.get $count) (i32.const 1)))))
   (local.set $j (i32.add (local.get $j) (i32.const 4)))
   (br $loop_p4)))
  (block $end_tail (loop $loop_tail
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
  (local.get $count))
