;; RECONSTRUCTED FROM BINARY, not recovered original source.
;; Input SHA-256: b7b68848eecdca210e2bf13288c5313ee38d47bb315046c40ff143654cb61656
;; Tool: wabt 1.0.39; readWasm(readDebugNames:true,simd:true), applyNames, toText.
;; Original comments, source formatting, authorship and license are NOT recovered.
(module
  (type (;0;) (func (param i32 i32 i32 i32 i32) (result i32)))
  (type (;1;) (func (param i32 i32 i32) (result i32)))
  (type (;2;) (func (param i32 i32 i32 i32 i32 i32)))
  (func (;0;) (type 0) (param i32 i32 i32 i32 i32) (result i32)
    (local i32 i32 i32 v128 v128 f64 f64 v128 v128 v128 v128 v128)
    i32.const 0
    local.set 5
    block  ;; label = @1
      loop  ;; label = @2
        local.get 5
        i32.const 7
        i32.add
        local.get 2
        i32.ge_u
        br_if 1 (;@1;)
        i32.const 0
        local.set 6
        v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
        local.set 12
        v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
        local.set 13
        v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
        local.set 14
        v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
        local.set 15
        block  ;; label = @3
          loop  ;; label = @4
            local.get 6
            local.get 3
            i32.ge_u
            br_if 1 (;@3;)
            local.get 0
            local.get 6
            local.get 2
            i32.mul
            i32.const 8
            i32.mul
            i32.add
            local.set 7
            local.get 7
            local.get 1
            i32.const 8
            i32.mul
            i32.add
            f64.load
            f64x2.splat
            local.set 16
            local.get 16
            local.get 7
            local.get 5
            i32.const 8
            i32.mul
            i32.add
            v128.load align=8
            f64x2.sub
            local.set 9
            local.get 12
            local.get 9
            local.get 9
            f64x2.mul
            f64x2.add
            local.set 12
            local.get 16
            local.get 7
            local.get 5
            i32.const 8
            i32.mul
            i32.add
            v128.load offset=16 align=8
            f64x2.sub
            local.set 9
            local.get 13
            local.get 9
            local.get 9
            f64x2.mul
            f64x2.add
            local.set 13
            local.get 16
            local.get 7
            local.get 5
            i32.const 8
            i32.mul
            i32.add
            v128.load offset=32 align=8
            f64x2.sub
            local.set 9
            local.get 14
            local.get 9
            local.get 9
            f64x2.mul
            f64x2.add
            local.set 14
            local.get 16
            local.get 7
            local.get 5
            i32.const 8
            i32.mul
            i32.add
            v128.load offset=48 align=8
            f64x2.sub
            local.set 9
            local.get 15
            local.get 9
            local.get 9
            f64x2.mul
            f64x2.add
            local.set 15
            local.get 6
            i32.const 1
            i32.add
            local.set 6
            br 0 (;@4;)
          end
        end
        local.get 12
        f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
        f64x2.splat
        f64x2.le
        i64x2.all_true
        i32.eqz
        if  ;; label = @3
          i32.const 1
          return
        end
        local.get 4
        local.get 5
        i32.const 8
        i32.mul
        i32.add
        local.get 12
        f64x2.sqrt
        v128.store align=8
        local.get 13
        f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
        f64x2.splat
        f64x2.le
        i64x2.all_true
        i32.eqz
        if  ;; label = @3
          i32.const 1
          return
        end
        local.get 4
        local.get 5
        i32.const 8
        i32.mul
        i32.add
        local.get 13
        f64x2.sqrt
        v128.store offset=16 align=8
        local.get 14
        f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
        f64x2.splat
        f64x2.le
        i64x2.all_true
        i32.eqz
        if  ;; label = @3
          i32.const 1
          return
        end
        local.get 4
        local.get 5
        i32.const 8
        i32.mul
        i32.add
        local.get 14
        f64x2.sqrt
        v128.store offset=32 align=8
        local.get 15
        f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
        f64x2.splat
        f64x2.le
        i64x2.all_true
        i32.eqz
        if  ;; label = @3
          i32.const 1
          return
        end
        local.get 4
        local.get 5
        i32.const 8
        i32.mul
        i32.add
        local.get 15
        f64x2.sqrt
        v128.store offset=48 align=8
        local.get 5
        i32.const 8
        i32.add
        local.set 5
        br 0 (;@2;)
      end
    end
    block  ;; label = @1
      loop  ;; label = @2
        local.get 5
        i32.const 1
        i32.add
        local.get 2
        i32.ge_u
        br_if 1 (;@1;)
        i32.const 0
        local.set 6
        v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
        local.set 8
        block  ;; label = @3
          loop  ;; label = @4
            local.get 6
            local.get 3
            i32.ge_u
            br_if 1 (;@3;)
            local.get 0
            local.get 6
            local.get 2
            i32.mul
            i32.const 8
            i32.mul
            i32.add
            local.set 7
            local.get 7
            local.get 1
            i32.const 8
            i32.mul
            i32.add
            f64.load
            f64x2.splat
            local.get 7
            local.get 5
            i32.const 8
            i32.mul
            i32.add
            v128.load align=8
            f64x2.sub
            local.set 9
            local.get 8
            local.get 9
            local.get 9
            f64x2.mul
            f64x2.add
            local.set 8
            local.get 6
            i32.const 1
            i32.add
            local.set 6
            br 0 (;@4;)
          end
        end
        local.get 8
        f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
        f64x2.splat
        f64x2.le
        i64x2.all_true
        i32.eqz
        if  ;; label = @3
          i32.const 1
          return
        end
        local.get 4
        local.get 5
        i32.const 8
        i32.mul
        i32.add
        local.get 8
        f64x2.sqrt
        v128.store align=8
        local.get 5
        i32.const 2
        i32.add
        local.set 5
        br 0 (;@2;)
      end
    end
    local.get 5
    local.get 2
    i32.lt_u
    if  ;; label = @1
      i32.const 0
      local.set 6
      f64.const 0x0p+0 (;=0;)
      local.set 10
      block  ;; label = @2
        loop  ;; label = @3
          local.get 6
          local.get 3
          i32.ge_u
          br_if 1 (;@2;)
          local.get 0
          local.get 6
          local.get 2
          i32.mul
          i32.const 8
          i32.mul
          i32.add
          local.set 7
          local.get 7
          local.get 1
          i32.const 8
          i32.mul
          i32.add
          f64.load
          local.get 7
          local.get 5
          i32.const 8
          i32.mul
          i32.add
          f64.load
          f64.sub
          local.set 11
          local.get 10
          local.get 11
          local.get 11
          f64.mul
          f64.add
          local.set 10
          local.get 6
          i32.const 1
          i32.add
          local.set 6
          br 0 (;@3;)
        end
      end
      local.get 10
      f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
      f64.gt
      if  ;; label = @2
        i32.const 1
        return
      end
      local.get 4
      local.get 5
      i32.const 8
      i32.mul
      i32.add
      local.get 10
      f64.sqrt
      f64.store
    end
    i32.const 0)
  (func (;1;) (type 1) (param i32 i32 i32) (result i32)
    (local i32 i32 f64 f64 v128 v128 v128)
    local.get 0
    local.get 2
    i32.const 8
    i32.mul
    i32.add
    f64.load
    local.set 5
    i32.const 1
    local.set 4
    block  ;; label = @1
      loop  ;; label = @2
        local.get 3
        i32.const 1
        i32.add
        local.get 1
        i32.ge_u
        br_if 1 (;@1;)
        local.get 0
        local.get 3
        i32.const 8
        i32.mul
        i32.add
        v128.load align=8
        local.set 7
        local.get 3
        i64.extend_i32_u
        i64x2.splat
        v128.const i32x4 0x00000000 0x00000000 0x00000001 0x00000000
        i64x2.add
        local.set 8
        local.get 7
        local.get 5
        f64x2.splat
        f64x2.lt
        local.get 7
        local.get 5
        f64x2.splat
        f64x2.eq
        local.get 8
        local.get 2
        i64.extend_i32_u
        i64x2.splat
        i64x2.lt_s
        v128.and
        v128.or
        local.set 9
        local.get 4
        local.get 9
        i64x2.bitmask
        i32.popcnt
        i32.add
        local.set 4
        local.get 3
        i32.const 2
        i32.add
        local.set 3
        br 0 (;@2;)
      end
    end
    local.get 3
    local.get 1
    i32.lt_u
    if  ;; label = @1
      local.get 0
      local.get 3
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set 6
      local.get 6
      local.get 5
      f64.lt
      local.get 6
      local.get 5
      f64.eq
      local.get 3
      local.get 2
      i32.lt_u
      i32.and
      i32.or
      if  ;; label = @2
        local.get 4
        i32.const 1
        i32.add
        local.set 4
      end
    end
    local.get 4)
  (func (;2;) (type 2) (param i32 i32 i32 i32 i32 i32)
    (local i32 i32 i32 i32 i32 i32 f64 f64)
    block  ;; label = @1
      loop  ;; label = @2
        local.get 6
        local.get 4
        i32.gt_u
        br_if 1 (;@1;)
        local.get 5
        local.get 6
        i32.const 4
        i32.mul
        i32.add
        i32.const 0
        i32.store
        local.get 6
        i32.const 1
        i32.add
        local.set 6
        br 0 (;@2;)
      end
    end
    i32.const 0
    local.set 6
    block  ;; label = @1
      loop  ;; label = @2
        local.get 6
        local.get 1
        i32.ge_u
        br_if 1 (;@1;)
        local.get 6
        local.get 2
        i32.ne
        if  ;; label = @3
          local.get 0
          local.get 6
          i32.const 8
          i32.mul
          i32.add
          f64.load
          local.set 12
          i32.const 0
          local.set 7
          local.get 4
          local.set 8
          block  ;; label = @4
            loop  ;; label = @5
              local.get 7
              local.get 8
              i32.ge_u
              br_if 1 (;@4;)
              local.get 7
              local.get 8
              i32.add
              i32.const 1
              i32.shr_u
              local.set 9
              local.get 3
              local.get 9
              i32.const 4
              i32.mul
              i32.add
              i32.load
              local.set 10
              local.get 0
              local.get 10
              i32.const 8
              i32.mul
              i32.add
              f64.load
              local.set 13
              local.get 13
              local.get 12
              f64.lt
              local.get 13
              local.get 12
              f64.eq
              local.get 10
              local.get 6
              i32.lt_u
              i32.and
              i32.or
              if  ;; label = @6
                local.get 9
                i32.const 1
                i32.add
                local.set 7
              else
                local.get 9
                local.set 8
              end
              br 0 (;@5;)
            end
          end
          local.get 5
          local.get 7
          i32.const 4
          i32.mul
          i32.add
          local.set 11
          local.get 11
          local.get 11
          i32.load
          i32.const 1
          i32.add
          i32.store
        end
        local.get 6
        i32.const 1
        i32.add
        local.set 6
        br 0 (;@2;)
      end
    end)
  (memory (;0;) 1)
  (export "memory" (memory 0))
  (export "row" (func 0))
  (export "rank" (func 1))
  (export "hist" (func 2)))

