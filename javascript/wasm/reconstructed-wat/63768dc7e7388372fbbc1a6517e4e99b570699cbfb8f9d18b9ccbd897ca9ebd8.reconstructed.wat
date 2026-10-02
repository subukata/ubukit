;; RECONSTRUCTED FROM BINARY, not recovered original source.
;; Input SHA-256: 63768dc7e7388372fbbc1a6517e4e99b570699cbfb8f9d18b9ccbd897ca9ebd8
;; Tool: wabt 1.0.39; readWasm(readDebugNames:true,simd:true), applyNames, toText.
;; Original comments, source formatting, authorship and license are NOT recovered.
(module
  (type (;0;) (func (param i32 i32 i32 i32 i32 f64 i32) (result i32)))
  (type (;1;) (func (param i32 i32 i32 i32 i32 i32 i32 i32 f64 f64) (result i32)))
  (type (;2;) (func (param f64 i32 i32 i32 i32 i32) (result i32)))
  (type (;3;) (func (param i32 i32 i32 i32 i32 i32 f64 i32 i32) (result i32)))
  (func (;0;) (type 0) (param $base i32) (param $i i32) (param $start i32) (param $end i32) (param $d i32) (param $cutoff f64) (param $output i32) (result i32)
    (local $j i32) (local $f i32) (local $io i32) (local $jo i32) (local $count i32) (local $sum v128) (local $diff v128) (local $sq f64) (local $tail f64)
    local.get $base
    local.get $i
    local.get $d
    i32.mul
    i32.const 8
    i32.mul
    i32.add
    local.set $io
    local.get $start
    local.set $j
    block  ;; label = @1
      loop  ;; label = @2
        local.get $j
        local.get $end
        i32.ge_u
        br_if 1 (;@1;)
        local.get $base
        local.get $j
        local.get $d
        i32.mul
        i32.const 8
        i32.mul
        i32.add
        local.set $jo
        i32.const 0
        local.set $f
        v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
        local.set $sum
        block  ;; label = @3
          loop  ;; label = @4
            local.get $f
            i32.const 1
            i32.add
            local.get $d
            i32.ge_u
            br_if 1 (;@3;)
            local.get $io
            local.get $f
            i32.const 8
            i32.mul
            i32.add
            v128.load align=8
            local.get $jo
            local.get $f
            i32.const 8
            i32.mul
            i32.add
            v128.load align=8
            f64x2.sub
            local.set $diff
            local.get $sum
            local.get $diff
            local.get $diff
            f64x2.mul
            f64x2.add
            local.set $sum
            local.get $f
            i32.const 2
            i32.add
            local.set $f
            br 0 (;@4;)
          end
        end
        local.get $sum
        f64x2.extract_lane 0
        local.get $sum
        f64x2.extract_lane 1
        f64.add
        local.set $sq
        local.get $f
        local.get $d
        i32.lt_u
        if  ;; label = @3
          local.get $io
          local.get $f
          i32.const 8
          i32.mul
          i32.add
          f64.load
          local.get $jo
          local.get $f
          i32.const 8
          i32.mul
          i32.add
          f64.load
          f64.sub
          local.set $tail
          local.get $sq
          local.get $tail
          local.get $tail
          f64.mul
          f64.add
          local.set $sq
        end
        local.get $sq
        local.get $cutoff
        f64.le
        if  ;; label = @3
          local.get $output
          local.get $count
          i32.const 4
          i32.mul
          i32.add
          local.get $j
          i32.store
          local.get $count
          i32.const 1
          i32.add
          local.set $count
        end
        local.get $j
        i32.const 1
        i32.add
        local.set $j
        br 0 (;@2;)
      end
    end
    local.get $count)
  (func (;1;) (type 1) (param $base i32) (param $d i32) (param $start i32) (param $end i32) (param $ptr i32) (param $ids i32) (param $degrees i32) (param $keep i32) (param $delta f64) (param $cutoff f64) (result i32)
    (local $i i32) (local $j i32) (local $q i32) (local $qe i32) (local $f i32) (local $io i32) (local $jo i32) (local $deg i32) (local $nonzero i32) (local $diff f64) (local $sum f64)
    local.get $start
    local.set $i
    block  ;; label = @1
      loop  ;; label = @2
        local.get $i
        local.get $end
        i32.ge_u
        br_if 1 (;@1;)
        local.get $base
        local.get $i
        local.get $d
        i32.mul
        i32.const 8
        i32.mul
        i32.add
        local.set $io
        local.get $degrees
        local.get $i
        i32.const 4
        i32.mul
        i32.add
        i32.load
        local.set $deg
        local.get $ptr
        local.get $i
        i32.const 4
        i32.mul
        i32.add
        i32.load
        local.set $q
        local.get $ptr
        local.get $i
        i32.const 4
        i32.mul
        i32.add
        i32.load offset=4
        local.set $qe
        block  ;; label = @3
          loop  ;; label = @4
            local.get $q
            local.get $qe
            i32.ge_u
            br_if 1 (;@3;)
            local.get $ids
            local.get $q
            i32.const 4
            i32.mul
            i32.add
            i32.load
            local.set $j
            block  ;; label = @5
              local.get $j
              local.get $i
              i32.le_u
              br_if 0 (;@5;)
              local.get $base
              local.get $j
              local.get $d
              i32.mul
              i32.const 8
              i32.mul
              i32.add
              local.set $jo
              i32.const 0
              local.set $f
              f64.const 0x0p+0 (;=0;)
              local.set $sum
              i32.const 0
              local.set $nonzero
              block  ;; label = @6
                loop  ;; label = @7
                  local.get $f
                  local.get $d
                  i32.ge_u
                  br_if 1 (;@6;)
                  local.get $io
                  local.get $f
                  i32.const 8
                  i32.mul
                  i32.add
                  f64.load
                  local.get $jo
                  local.get $f
                  i32.const 8
                  i32.mul
                  i32.add
                  f64.load
                  f64.sub
                  local.set $diff
                  local.get $nonzero
                  local.get $diff
                  f64.const 0x0p+0 (;=0;)
                  f64.ne
                  i32.or
                  local.set $nonzero
                  local.get $delta
                  f64.const 0x0p+0 (;=0;)
                  f64.eq
                  local.get $nonzero
                  i32.and
                  br_if 2 (;@5;)
                  local.get $sum
                  local.get $diff
                  local.get $diff
                  f64.mul
                  f64.add
                  local.set $sum
                  local.get $sum
                  local.get $cutoff
                  f64.gt
                  br_if 2 (;@5;)
                  local.get $f
                  i32.const 1
                  i32.add
                  local.set $f
                  br 0 (;@7;)
                end
              end
              local.get $sum
              f64.const 0x0p+0 (;=0;)
              f64.eq
              local.get $nonzero
              i32.and
              if  ;; label = @6
                i32.const -1
                return
              end
              local.get $sum
              f64.const 0x0p+0 (;=0;)
              f64.gt
              local.get $sum
              f64.const 0x1p-1022 (;=2.22507e-308;)
              f64.lt
              i32.and
              if  ;; label = @6
                i32.const -2
                return
              end
              local.get $sum
              f64.sqrt
              local.get $delta
              f64.le
              if  ;; label = @6
                local.get $keep
                local.get $q
                i32.add
                i32.const 1
                i32.store8
                local.get $deg
                i32.const 1
                i32.add
                local.set $deg
                local.get $degrees
                local.get $j
                i32.const 4
                i32.mul
                i32.add
                local.get $degrees
                local.get $j
                i32.const 4
                i32.mul
                i32.add
                i32.load
                i32.const 1
                i32.add
                i32.store
              end
            end
            local.get $q
            i32.const 1
            i32.add
            local.set $q
            br 0 (;@4;)
          end
        end
        local.get $degrees
        local.get $i
        i32.const 4
        i32.mul
        i32.add
        local.get $deg
        i32.store
        local.get $i
        i32.const 1
        i32.add
        local.set $i
        br 0 (;@2;)
      end
    end
    i32.const 0)
  (func $check_tiny (type 2) (param $sq f64) (param $base i32) (param $i i32) (param $j i32) (param $n i32) (param $d i32) (result i32)
    (local $f i32) (local $off i32)
    local.get $sq
    f64.const 0x0p+0 (;=0;)
    f64.gt
    if  ;; label = @1
      i32.const -2
      return
    end
    block  ;; label = @1
      loop  ;; label = @2
        local.get $f
        local.get $d
        i32.ge_u
        br_if 1 (;@1;)
        local.get $base
        local.get $f
        local.get $n
        i32.mul
        i32.const 8
        i32.mul
        i32.add
        local.set $off
        local.get $off
        local.get $i
        i32.const 8
        i32.mul
        i32.add
        f64.load
        local.get $off
        local.get $j
        i32.const 8
        i32.mul
        i32.add
        f64.load
        f64.ne
        if  ;; label = @3
          i32.const -1
          return
        end
        local.get $f
        i32.const 1
        i32.add
        local.set $f
        br 0 (;@2;)
      end
    end
    i32.const 0)
  (func (;3;) (type 3) (param $base i32) (param $i i32) (param $n i32) (param $d i32) (param $start i32) (param $end i32) (param $delta f64) (param $out i32) (param $error i32) (result i32)
    (local $j i32) (local $f i32) (local $off i32) (local $count i32) (local $mask i32) (local $small i32) (local $code i32) (local $s0 v128) (local $s1 v128) (local $s2 v128) (local $s3 v128) (local $query v128) (local $diff v128) (local $sq f64) (local $tail f64)
    local.get $error
    i32.const 0
    i32.store
    local.get $start
    local.set $j
    block  ;; label = @1
      loop  ;; label = @2
        local.get $j
        i32.const 7
        i32.add
        local.get $end
        i32.ge_u
        br_if 1 (;@1;)
        i32.const 0
        local.set $f
        v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
        local.set $s0
        v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
        local.set $s1
        v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
        local.set $s2
        v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
        local.set $s3
        block  ;; label = @3
          loop  ;; label = @4
            local.get $f
            local.get $d
            i32.ge_u
            br_if 1 (;@3;)
            local.get $base
            local.get $f
            local.get $n
            i32.mul
            i32.const 8
            i32.mul
            i32.add
            local.set $off
            local.get $off
            local.get $i
            i32.const 8
            i32.mul
            i32.add
            f64.load
            f64x2.splat
            local.set $query
            local.get $query
            local.get $off
            local.get $j
            i32.const 8
            i32.mul
            i32.add
            v128.load align=8
            f64x2.sub
            local.set $diff
            local.get $s0
            local.get $diff
            local.get $diff
            f64x2.mul
            f64x2.add
            local.set $s0
            local.get $query
            local.get $off
            local.get $j
            i32.const 8
            i32.mul
            i32.add
            v128.load offset=16 align=8
            f64x2.sub
            local.set $diff
            local.get $s1
            local.get $diff
            local.get $diff
            f64x2.mul
            f64x2.add
            local.set $s1
            local.get $query
            local.get $off
            local.get $j
            i32.const 8
            i32.mul
            i32.add
            v128.load offset=32 align=8
            f64x2.sub
            local.set $diff
            local.get $s2
            local.get $diff
            local.get $diff
            f64x2.mul
            f64x2.add
            local.set $s2
            local.get $query
            local.get $off
            local.get $j
            i32.const 8
            i32.mul
            i32.add
            v128.load offset=48 align=8
            f64x2.sub
            local.set $diff
            local.get $s3
            local.get $diff
            local.get $diff
            f64x2.mul
            f64x2.add
            local.set $s3
            local.get $f
            i32.const 1
            i32.add
            local.set $f
            br 0 (;@4;)
          end
        end
        local.get $s0
        f64x2.sqrt
        local.get $delta
        f64x2.splat
        f64x2.le
        i64x2.bitmask
        local.set $mask
        local.get $s0
        f64.const 0x1p-1022 (;=2.22507e-308;)
        f64x2.splat
        f64x2.lt
        i64x2.bitmask
        local.set $small
        local.get $small
        if  ;; label = @3
          local.get $small
          i32.const 1
          i32.and
          if  ;; label = @4
            local.get $s0
            f64x2.extract_lane 0
            local.get $base
            local.get $i
            local.get $j
            i32.const 0
            i32.add
            local.get $n
            local.get $d
            call $check_tiny
            local.set $code
            local.get $code
            if  ;; label = @5
              local.get $error
              local.get $code
              i32.store
              local.get $count
              return
            end
          end
          local.get $mask
          i32.const 1
          i32.and
          if  ;; label = @4
            local.get $out
            local.get $count
            i32.const 4
            i32.mul
            i32.add
            local.get $j
            i32.const 0
            i32.add
            i32.store
            local.get $count
            i32.const 1
            i32.add
            local.set $count
          end
          local.get $small
          i32.const 2
          i32.and
          if  ;; label = @4
            local.get $s0
            f64x2.extract_lane 1
            local.get $base
            local.get $i
            local.get $j
            i32.const 1
            i32.add
            local.get $n
            local.get $d
            call $check_tiny
            local.set $code
            local.get $code
            if  ;; label = @5
              local.get $error
              local.get $code
              i32.store
              local.get $count
              return
            end
          end
          local.get $mask
          i32.const 2
          i32.and
          if  ;; label = @4
            local.get $out
            local.get $count
            i32.const 4
            i32.mul
            i32.add
            local.get $j
            i32.const 1
            i32.add
            i32.store
            local.get $count
            i32.const 1
            i32.add
            local.set $count
          end
        else
          local.get $mask
          i32.const 1
          i32.and
          if  ;; label = @4
            local.get $out
            local.get $count
            i32.const 4
            i32.mul
            i32.add
            local.get $j
            i32.const 0
            i32.add
            i32.store
            local.get $count
            i32.const 1
            i32.add
            local.set $count
          end
          local.get $mask
          i32.const 2
          i32.and
          if  ;; label = @4
            local.get $out
            local.get $count
            i32.const 4
            i32.mul
            i32.add
            local.get $j
            i32.const 1
            i32.add
            i32.store
            local.get $count
            i32.const 1
            i32.add
            local.set $count
          end
        end
        local.get $s1
        f64x2.sqrt
        local.get $delta
        f64x2.splat
        f64x2.le
        i64x2.bitmask
        local.set $mask
        local.get $s1
        f64.const 0x1p-1022 (;=2.22507e-308;)
        f64x2.splat
        f64x2.lt
        i64x2.bitmask
        local.set $small
        local.get $small
        if  ;; label = @3
          local.get $small
          i32.const 1
          i32.and
          if  ;; label = @4
            local.get $s1
            f64x2.extract_lane 0
            local.get $base
            local.get $i
            local.get $j
            i32.const 2
            i32.add
            local.get $n
            local.get $d
            call $check_tiny
            local.set $code
            local.get $code
            if  ;; label = @5
              local.get $error
              local.get $code
              i32.store
              local.get $count
              return
            end
          end
          local.get $mask
          i32.const 1
          i32.and
          if  ;; label = @4
            local.get $out
            local.get $count
            i32.const 4
            i32.mul
            i32.add
            local.get $j
            i32.const 2
            i32.add
            i32.store
            local.get $count
            i32.const 1
            i32.add
            local.set $count
          end
          local.get $small
          i32.const 2
          i32.and
          if  ;; label = @4
            local.get $s1
            f64x2.extract_lane 1
            local.get $base
            local.get $i
            local.get $j
            i32.const 3
            i32.add
            local.get $n
            local.get $d
            call $check_tiny
            local.set $code
            local.get $code
            if  ;; label = @5
              local.get $error
              local.get $code
              i32.store
              local.get $count
              return
            end
          end
          local.get $mask
          i32.const 2
          i32.and
          if  ;; label = @4
            local.get $out
            local.get $count
            i32.const 4
            i32.mul
            i32.add
            local.get $j
            i32.const 3
            i32.add
            i32.store
            local.get $count
            i32.const 1
            i32.add
            local.set $count
          end
        else
          local.get $mask
          i32.const 1
          i32.and
          if  ;; label = @4
            local.get $out
            local.get $count
            i32.const 4
            i32.mul
            i32.add
            local.get $j
            i32.const 2
            i32.add
            i32.store
            local.get $count
            i32.const 1
            i32.add
            local.set $count
          end
          local.get $mask
          i32.const 2
          i32.and
          if  ;; label = @4
            local.get $out
            local.get $count
            i32.const 4
            i32.mul
            i32.add
            local.get $j
            i32.const 3
            i32.add
            i32.store
            local.get $count
            i32.const 1
            i32.add
            local.set $count
          end
        end
        local.get $s2
        f64x2.sqrt
        local.get $delta
        f64x2.splat
        f64x2.le
        i64x2.bitmask
        local.set $mask
        local.get $s2
        f64.const 0x1p-1022 (;=2.22507e-308;)
        f64x2.splat
        f64x2.lt
        i64x2.bitmask
        local.set $small
        local.get $small
        if  ;; label = @3
          local.get $small
          i32.const 1
          i32.and
          if  ;; label = @4
            local.get $s2
            f64x2.extract_lane 0
            local.get $base
            local.get $i
            local.get $j
            i32.const 4
            i32.add
            local.get $n
            local.get $d
            call $check_tiny
            local.set $code
            local.get $code
            if  ;; label = @5
              local.get $error
              local.get $code
              i32.store
              local.get $count
              return
            end
          end
          local.get $mask
          i32.const 1
          i32.and
          if  ;; label = @4
            local.get $out
            local.get $count
            i32.const 4
            i32.mul
            i32.add
            local.get $j
            i32.const 4
            i32.add
            i32.store
            local.get $count
            i32.const 1
            i32.add
            local.set $count
          end
          local.get $small
          i32.const 2
          i32.and
          if  ;; label = @4
            local.get $s2
            f64x2.extract_lane 1
            local.get $base
            local.get $i
            local.get $j
            i32.const 5
            i32.add
            local.get $n
            local.get $d
            call $check_tiny
            local.set $code
            local.get $code
            if  ;; label = @5
              local.get $error
              local.get $code
              i32.store
              local.get $count
              return
            end
          end
          local.get $mask
          i32.const 2
          i32.and
          if  ;; label = @4
            local.get $out
            local.get $count
            i32.const 4
            i32.mul
            i32.add
            local.get $j
            i32.const 5
            i32.add
            i32.store
            local.get $count
            i32.const 1
            i32.add
            local.set $count
          end
        else
          local.get $mask
          i32.const 1
          i32.and
          if  ;; label = @4
            local.get $out
            local.get $count
            i32.const 4
            i32.mul
            i32.add
            local.get $j
            i32.const 4
            i32.add
            i32.store
            local.get $count
            i32.const 1
            i32.add
            local.set $count
          end
          local.get $mask
          i32.const 2
          i32.and
          if  ;; label = @4
            local.get $out
            local.get $count
            i32.const 4
            i32.mul
            i32.add
            local.get $j
            i32.const 5
            i32.add
            i32.store
            local.get $count
            i32.const 1
            i32.add
            local.set $count
          end
        end
        local.get $s3
        f64x2.sqrt
        local.get $delta
        f64x2.splat
        f64x2.le
        i64x2.bitmask
        local.set $mask
        local.get $s3
        f64.const 0x1p-1022 (;=2.22507e-308;)
        f64x2.splat
        f64x2.lt
        i64x2.bitmask
        local.set $small
        local.get $small
        if  ;; label = @3
          local.get $small
          i32.const 1
          i32.and
          if  ;; label = @4
            local.get $s3
            f64x2.extract_lane 0
            local.get $base
            local.get $i
            local.get $j
            i32.const 6
            i32.add
            local.get $n
            local.get $d
            call $check_tiny
            local.set $code
            local.get $code
            if  ;; label = @5
              local.get $error
              local.get $code
              i32.store
              local.get $count
              return
            end
          end
          local.get $mask
          i32.const 1
          i32.and
          if  ;; label = @4
            local.get $out
            local.get $count
            i32.const 4
            i32.mul
            i32.add
            local.get $j
            i32.const 6
            i32.add
            i32.store
            local.get $count
            i32.const 1
            i32.add
            local.set $count
          end
          local.get $small
          i32.const 2
          i32.and
          if  ;; label = @4
            local.get $s3
            f64x2.extract_lane 1
            local.get $base
            local.get $i
            local.get $j
            i32.const 7
            i32.add
            local.get $n
            local.get $d
            call $check_tiny
            local.set $code
            local.get $code
            if  ;; label = @5
              local.get $error
              local.get $code
              i32.store
              local.get $count
              return
            end
          end
          local.get $mask
          i32.const 2
          i32.and
          if  ;; label = @4
            local.get $out
            local.get $count
            i32.const 4
            i32.mul
            i32.add
            local.get $j
            i32.const 7
            i32.add
            i32.store
            local.get $count
            i32.const 1
            i32.add
            local.set $count
          end
        else
          local.get $mask
          i32.const 1
          i32.and
          if  ;; label = @4
            local.get $out
            local.get $count
            i32.const 4
            i32.mul
            i32.add
            local.get $j
            i32.const 6
            i32.add
            i32.store
            local.get $count
            i32.const 1
            i32.add
            local.set $count
          end
          local.get $mask
          i32.const 2
          i32.and
          if  ;; label = @4
            local.get $out
            local.get $count
            i32.const 4
            i32.mul
            i32.add
            local.get $j
            i32.const 7
            i32.add
            i32.store
            local.get $count
            i32.const 1
            i32.add
            local.set $count
          end
        end
        local.get $j
        i32.const 8
        i32.add
        local.set $j
        br 0 (;@2;)
      end
    end
    block  ;; label = @1
      loop  ;; label = @2
        local.get $j
        i32.const 1
        i32.add
        local.get $end
        i32.ge_u
        br_if 1 (;@1;)
        i32.const 0
        local.set $f
        v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
        local.set $s0
        block  ;; label = @3
          loop  ;; label = @4
            local.get $f
            local.get $d
            i32.ge_u
            br_if 1 (;@3;)
            local.get $base
            local.get $f
            local.get $n
            i32.mul
            i32.const 8
            i32.mul
            i32.add
            local.set $off
            local.get $off
            local.get $i
            i32.const 8
            i32.mul
            i32.add
            f64.load
            f64x2.splat
            local.set $query
            local.get $query
            local.get $off
            local.get $j
            i32.const 8
            i32.mul
            i32.add
            v128.load align=8
            f64x2.sub
            local.set $diff
            local.get $s0
            local.get $diff
            local.get $diff
            f64x2.mul
            f64x2.add
            local.set $s0
            local.get $f
            i32.const 1
            i32.add
            local.set $f
            br 0 (;@4;)
          end
        end
        local.get $s0
        f64x2.sqrt
        local.get $delta
        f64x2.splat
        f64x2.le
        i64x2.bitmask
        local.set $mask
        local.get $s0
        f64.const 0x1p-1022 (;=2.22507e-308;)
        f64x2.splat
        f64x2.lt
        i64x2.bitmask
        local.set $small
        local.get $small
        if  ;; label = @3
          local.get $small
          i32.const 1
          i32.and
          if  ;; label = @4
            local.get $s0
            f64x2.extract_lane 0
            local.get $base
            local.get $i
            local.get $j
            i32.const 0
            i32.add
            local.get $n
            local.get $d
            call $check_tiny
            local.set $code
            local.get $code
            if  ;; label = @5
              local.get $error
              local.get $code
              i32.store
              local.get $count
              return
            end
          end
          local.get $mask
          i32.const 1
          i32.and
          if  ;; label = @4
            local.get $out
            local.get $count
            i32.const 4
            i32.mul
            i32.add
            local.get $j
            i32.const 0
            i32.add
            i32.store
            local.get $count
            i32.const 1
            i32.add
            local.set $count
          end
          local.get $small
          i32.const 2
          i32.and
          if  ;; label = @4
            local.get $s0
            f64x2.extract_lane 1
            local.get $base
            local.get $i
            local.get $j
            i32.const 1
            i32.add
            local.get $n
            local.get $d
            call $check_tiny
            local.set $code
            local.get $code
            if  ;; label = @5
              local.get $error
              local.get $code
              i32.store
              local.get $count
              return
            end
          end
          local.get $mask
          i32.const 2
          i32.and
          if  ;; label = @4
            local.get $out
            local.get $count
            i32.const 4
            i32.mul
            i32.add
            local.get $j
            i32.const 1
            i32.add
            i32.store
            local.get $count
            i32.const 1
            i32.add
            local.set $count
          end
        else
          local.get $mask
          i32.const 1
          i32.and
          if  ;; label = @4
            local.get $out
            local.get $count
            i32.const 4
            i32.mul
            i32.add
            local.get $j
            i32.const 0
            i32.add
            i32.store
            local.get $count
            i32.const 1
            i32.add
            local.set $count
          end
          local.get $mask
          i32.const 2
          i32.and
          if  ;; label = @4
            local.get $out
            local.get $count
            i32.const 4
            i32.mul
            i32.add
            local.get $j
            i32.const 1
            i32.add
            i32.store
            local.get $count
            i32.const 1
            i32.add
            local.set $count
          end
        end
        local.get $j
        i32.const 2
        i32.add
        local.set $j
        br 0 (;@2;)
      end
    end
    local.get $j
    local.get $end
    i32.lt_u
    if  ;; label = @1
      i32.const 0
      local.set $f
      f64.const 0x0p+0 (;=0;)
      local.set $sq
      block  ;; label = @2
        loop  ;; label = @3
          local.get $f
          local.get $d
          i32.ge_u
          br_if 1 (;@2;)
          local.get $base
          local.get $f
          local.get $n
          i32.mul
          i32.const 8
          i32.mul
          i32.add
          local.set $off
          local.get $off
          local.get $i
          i32.const 8
          i32.mul
          i32.add
          f64.load
          local.get $off
          local.get $j
          i32.const 8
          i32.mul
          i32.add
          f64.load
          f64.sub
          local.set $tail
          local.get $sq
          local.get $tail
          local.get $tail
          f64.mul
          f64.add
          local.set $sq
          local.get $f
          i32.const 1
          i32.add
          local.set $f
          br 0 (;@3;)
        end
      end
      local.get $sq
      f64.const 0x1p-1022 (;=2.22507e-308;)
      f64.lt
      if  ;; label = @2
        local.get $sq
        local.get $base
        local.get $i
        local.get $j
        local.get $n
        local.get $d
        call $check_tiny
        local.set $code
        local.get $code
        if  ;; label = @3
          local.get $error
          local.get $code
          i32.store
          local.get $count
          return
        end
      end
      local.get $sq
      f64.sqrt
      local.get $delta
      f64.le
      if  ;; label = @2
        local.get $out
        local.get $count
        i32.const 4
        i32.mul
        i32.add
        local.get $j
        i32.store
        local.get $count
        i32.const 1
        i32.add
        local.set $count
      end
    end
    local.get $count)
  (memory (;0;) 1)
  (export "memory" (memory 0))
  (export "scan" (func 0))
  (export "filter" (func 1))
  (export "scan_exact" (func 3)))

