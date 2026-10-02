;; RECONSTRUCTED FROM BINARY, not recovered original source.
;; Input SHA-256: eaa97f76b4b7d0c1847e8f157f7b53df06c12379050c4d31f50e9316db78f37d
;; Tool: wabt 1.0.39; readWasm(readDebugNames:true,simd:true), applyNames, toText.
;; Original comments, source formatting, authorship and license are NOT recovered.
(module
  (type (;0;) (func (param i32 i32 i32 i32 i32 i32 i32 i32 i32)))
  (func (;0;) (type 0) (param i32 i32 i32 i32 i32 i32 i32 i32 i32)
    (local i32 i32 i32 i32 i32 i32 v128 v128 v128 f64 f64 f64)
    block  ;; label = @1
      loop  ;; label = @2
        local.get 9
        local.get 2
        i32.ge_u
        br_if 1 (;@1;)
        local.get 0
        local.get 9
        local.get 3
        i32.mul
        i32.const 8
        i32.mul
        i32.add
        local.set 14
        f64.const inf (;=inf;)
        local.set 20
        i32.const 0
        local.set 12
        i32.const 0
        local.set 13
        i32.const 0
        local.set 10
        block  ;; label = @3
          loop  ;; label = @4
            local.get 10
            local.get 4
            i32.ge_u
            br_if 1 (;@3;)
            v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
            local.set 15
            i32.const 0
            local.set 11
            block  ;; label = @5
              loop  ;; label = @6
                local.get 11
                local.get 3
                i32.ge_u
                br_if 1 (;@5;)
                local.get 14
                local.get 11
                i32.const 8
                i32.mul
                i32.add
                f64.load
                f64x2.splat
                local.set 17
                local.get 17
                local.get 1
                local.get 11
                local.get 5
                i32.mul
                local.get 10
                i32.add
                i32.const 8
                i32.mul
                i32.add
                v128.load
                f64x2.sub
                local.set 16
                local.get 15
                local.get 16
                local.get 16
                f64x2.mul
                f64x2.add
                local.set 15
                local.get 11
                i32.const 1
                i32.add
                local.set 11
                br 0 (;@6;)
              end
            end
            local.get 15
            f64x2.extract_lane 0
            local.set 18
            local.get 18
            local.get 18
            f64.ne
            local.get 18
            f64.const 0x1p-1022 (;=2.22507e-308;)
            f64.lt
            local.get 18
            f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
            f64.gt
            i32.or
            i32.or
            if  ;; label = @5
              i32.const 1
              local.set 13
            end
            local.get 18
            local.get 20
            f64.lt
            if  ;; label = @5
              local.get 18
              local.set 20
              local.get 10
              local.set 12
            end
            local.get 10
            i32.const 1
            i32.add
            local.get 4
            i32.lt_u
            if  ;; label = @5
              local.get 15
              f64x2.extract_lane 1
              local.set 19
              local.get 19
              local.get 19
              f64.ne
              local.get 19
              f64.const 0x1p-1022 (;=2.22507e-308;)
              f64.lt
              local.get 19
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.gt
              i32.or
              i32.or
              if  ;; label = @6
                i32.const 1
                local.set 13
              end
              local.get 19
              local.get 20
              f64.lt
              if  ;; label = @6
                local.get 19
                local.set 20
                local.get 10
                i32.const 1
                i32.add
                local.set 12
              end
            end
            local.get 10
            i32.const 2
            i32.add
            local.set 10
            br 0 (;@4;)
          end
        end
        local.get 6
        local.get 9
        i32.const 4
        i32.mul
        i32.add
        local.get 12
        i32.store
        local.get 7
        local.get 9
        i32.const 8
        i32.mul
        i32.add
        local.get 20
        f64.store
        local.get 8
        local.get 9
        i32.const 4
        i32.mul
        i32.add
        local.get 13
        i32.store
        local.get 9
        i32.const 1
        i32.add
        local.set 9
        br 0 (;@2;)
      end
    end)
  (memory (;0;) 1)
  (export "memory" (memory 0))
  (export "nearest" (func 0)))

