;; RECONSTRUCTED FROM BINARY, not recovered original source.
;; Input SHA-256: 3cc6399d0c0a190daf937e3a319e7a54d5a334d26f87f6672d07ad74949d456c
;; Tool: wabt 1.0.39; readWasm(readDebugNames:true,simd:true), applyNames, toText.
;; Original comments, source formatting, authorship and license are NOT recovered.
(module
  (type (;0;) (func (param i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)))
  (type (;1;) (func (param i32 i32 i32 i32 i32 i32 i32 i32 f64) (result i32)))
  (type (;2;) (func (param i32 i32 i32 i32 i32 i32 i32 i32 f64)))
  (func (;0;) (type 0) (param i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
    (local i32 i32 i32 i32 i32 i32 i32 i32 f64 v128 i32)
    block  ;; label = @1
      loop  ;; label = @2
        local.get 10
        local.get 6
        i32.ge_u
        br_if 1 (;@1;)
        local.get 0
        local.get 10
        local.get 7
        i32.mul
        i32.const 8
        i32.mul
        i32.add
        local.set 14
        local.get 5
        local.get 10
        local.get 9
        i32.mul
        i32.const 8
        i32.mul
        i32.add
        local.set 16
        i32.const 0
        local.set 11
        block  ;; label = @3
          loop  ;; label = @4
            local.get 11
            local.get 8
            i32.ge_u
            br_if 1 (;@3;)
            local.get 1
            local.get 10
            local.get 8
            i32.mul
            local.get 11
            i32.add
            i32.const 8
            i32.mul
            i32.add
            f64.load
            local.set 18
            local.get 18
            f64x2.splat
            local.set 19
            local.get 4
            local.get 11
            i32.const 8
            i32.mul
            i32.add
            local.set 20
            local.get 20
            local.get 20
            f64.load
            local.get 18
            f64.add
            f64.store
            local.get 3
            local.get 11
            local.get 7
            i32.mul
            i32.const 8
            i32.mul
            i32.add
            local.set 15
            local.get 2
            local.get 11
            local.get 9
            i32.mul
            i32.const 8
            i32.mul
            i32.add
            local.set 17
            i32.const 0
            local.set 12
            block  ;; label = @5
              loop  ;; label = @6
                local.get 12
                i32.const 1
                i32.add
                local.get 7
                i32.ge_u
                br_if 1 (;@5;)
                local.get 15
                local.get 12
                i32.const 8
                i32.mul
                i32.add
                local.set 20
                local.get 20
                local.get 20
                v128.load
                local.get 19
                local.get 14
                local.get 12
                i32.const 8
                i32.mul
                i32.add
                v128.load
                f64x2.mul
                f64x2.add
                v128.store
                local.get 12
                i32.const 2
                i32.add
                local.set 12
                br 0 (;@6;)
              end
            end
            local.get 12
            local.get 7
            i32.lt_u
            if  ;; label = @5
              local.get 15
              local.get 12
              i32.const 8
              i32.mul
              i32.add
              local.set 20
              local.get 20
              local.get 20
              f64.load
              local.get 18
              local.get 14
              local.get 12
              i32.const 8
              i32.mul
              i32.add
              f64.load
              f64.mul
              f64.add
              f64.store
            end
            i32.const 0
            local.set 13
            block  ;; label = @5
              loop  ;; label = @6
                local.get 13
                i32.const 1
                i32.add
                local.get 9
                i32.ge_u
                br_if 1 (;@5;)
                local.get 16
                local.get 13
                i32.const 8
                i32.mul
                i32.add
                local.set 20
                local.get 20
                local.get 20
                v128.load
                local.get 19
                local.get 17
                local.get 13
                i32.const 8
                i32.mul
                i32.add
                v128.load
                f64x2.mul
                f64x2.add
                v128.store
                local.get 13
                i32.const 2
                i32.add
                local.set 13
                br 0 (;@6;)
              end
            end
            local.get 13
            local.get 9
            i32.lt_u
            if  ;; label = @5
              local.get 16
              local.get 13
              i32.const 8
              i32.mul
              i32.add
              local.set 20
              local.get 20
              local.get 20
              f64.load
              local.get 18
              local.get 17
              local.get 13
              i32.const 8
              i32.mul
              i32.add
              f64.load
              f64.mul
              f64.add
              f64.store
            end
            local.get 11
            i32.const 1
            i32.add
            local.set 11
            br 0 (;@4;)
          end
        end
        local.get 10
        i32.const 1
        i32.add
        local.set 10
        br 0 (;@2;)
      end
    end)
  (func (;1;) (type 0) (param i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
    (local i32 i32 i32 i32 i32 i32 i32 v128 v128 f64 f64 f64 v128 i32 i32 f64 v128 i32 i32 f64 v128 i32 i32 f64 v128 i32 i32)
    block  ;; label = @1
      loop  ;; label = @2
        local.get 10
        local.get 6
        i32.ge_u
        br_if 1 (;@1;)
        local.get 0
        local.get 10
        local.get 7
        i32.mul
        i32.const 8
        i32.mul
        i32.add
        local.set 14
        local.get 5
        local.get 10
        local.get 9
        i32.mul
        i32.const 8
        i32.mul
        i32.add
        local.set 15
        i32.const 0
        local.set 11
        block  ;; label = @3
          loop  ;; label = @4
            local.get 11
            i32.const 3
            i32.add
            local.get 8
            i32.ge_u
            br_if 1 (;@3;)
            local.get 1
            local.get 10
            local.get 8
            i32.mul
            local.get 11
            i32.const 0
            i32.add
            i32.add
            i32.const 8
            i32.mul
            i32.add
            f64.load
            local.set 21
            local.get 21
            f64x2.splat
            local.set 22
            local.get 4
            local.get 11
            i32.const 0
            i32.add
            i32.const 8
            i32.mul
            i32.add
            local.set 16
            local.get 16
            local.get 16
            f64.load
            local.get 21
            f64.add
            f64.store
            local.get 3
            local.get 11
            i32.const 0
            i32.add
            local.get 7
            i32.mul
            i32.const 8
            i32.mul
            i32.add
            local.set 23
            local.get 2
            local.get 11
            i32.const 0
            i32.add
            local.get 9
            i32.mul
            i32.const 8
            i32.mul
            i32.add
            local.set 24
            local.get 1
            local.get 10
            local.get 8
            i32.mul
            local.get 11
            i32.const 1
            i32.add
            i32.add
            i32.const 8
            i32.mul
            i32.add
            f64.load
            local.set 25
            local.get 25
            f64x2.splat
            local.set 26
            local.get 4
            local.get 11
            i32.const 1
            i32.add
            i32.const 8
            i32.mul
            i32.add
            local.set 16
            local.get 16
            local.get 16
            f64.load
            local.get 25
            f64.add
            f64.store
            local.get 3
            local.get 11
            i32.const 1
            i32.add
            local.get 7
            i32.mul
            i32.const 8
            i32.mul
            i32.add
            local.set 27
            local.get 2
            local.get 11
            i32.const 1
            i32.add
            local.get 9
            i32.mul
            i32.const 8
            i32.mul
            i32.add
            local.set 28
            local.get 1
            local.get 10
            local.get 8
            i32.mul
            local.get 11
            i32.const 2
            i32.add
            i32.add
            i32.const 8
            i32.mul
            i32.add
            f64.load
            local.set 29
            local.get 29
            f64x2.splat
            local.set 30
            local.get 4
            local.get 11
            i32.const 2
            i32.add
            i32.const 8
            i32.mul
            i32.add
            local.set 16
            local.get 16
            local.get 16
            f64.load
            local.get 29
            f64.add
            f64.store
            local.get 3
            local.get 11
            i32.const 2
            i32.add
            local.get 7
            i32.mul
            i32.const 8
            i32.mul
            i32.add
            local.set 31
            local.get 2
            local.get 11
            i32.const 2
            i32.add
            local.get 9
            i32.mul
            i32.const 8
            i32.mul
            i32.add
            local.set 32
            local.get 1
            local.get 10
            local.get 8
            i32.mul
            local.get 11
            i32.const 3
            i32.add
            i32.add
            i32.const 8
            i32.mul
            i32.add
            f64.load
            local.set 33
            local.get 33
            f64x2.splat
            local.set 34
            local.get 4
            local.get 11
            i32.const 3
            i32.add
            i32.const 8
            i32.mul
            i32.add
            local.set 16
            local.get 16
            local.get 16
            f64.load
            local.get 33
            f64.add
            f64.store
            local.get 3
            local.get 11
            i32.const 3
            i32.add
            local.get 7
            i32.mul
            i32.const 8
            i32.mul
            i32.add
            local.set 35
            local.get 2
            local.get 11
            i32.const 3
            i32.add
            local.get 9
            i32.mul
            i32.const 8
            i32.mul
            i32.add
            local.set 36
            i32.const 0
            local.set 12
            block  ;; label = @5
              loop  ;; label = @6
                local.get 12
                i32.const 1
                i32.add
                local.get 7
                i32.ge_u
                br_if 1 (;@5;)
                local.get 14
                local.get 12
                i32.const 8
                i32.mul
                i32.add
                v128.load
                local.set 17
                local.get 23
                local.get 12
                i32.const 8
                i32.mul
                i32.add
                local.set 16
                local.get 16
                local.get 16
                v128.load
                local.get 22
                local.get 17
                f64x2.mul
                f64x2.add
                v128.store
                local.get 27
                local.get 12
                i32.const 8
                i32.mul
                i32.add
                local.set 16
                local.get 16
                local.get 16
                v128.load
                local.get 26
                local.get 17
                f64x2.mul
                f64x2.add
                v128.store
                local.get 31
                local.get 12
                i32.const 8
                i32.mul
                i32.add
                local.set 16
                local.get 16
                local.get 16
                v128.load
                local.get 30
                local.get 17
                f64x2.mul
                f64x2.add
                v128.store
                local.get 35
                local.get 12
                i32.const 8
                i32.mul
                i32.add
                local.set 16
                local.get 16
                local.get 16
                v128.load
                local.get 34
                local.get 17
                f64x2.mul
                f64x2.add
                v128.store
                local.get 12
                i32.const 2
                i32.add
                local.set 12
                br 0 (;@6;)
              end
            end
            local.get 12
            local.get 7
            i32.lt_u
            if  ;; label = @5
              local.get 14
              local.get 12
              i32.const 8
              i32.mul
              i32.add
              f64.load
              local.set 19
              local.get 23
              local.get 12
              i32.const 8
              i32.mul
              i32.add
              local.set 16
              local.get 16
              local.get 16
              f64.load
              local.get 21
              local.get 19
              f64.mul
              f64.add
              f64.store
              local.get 27
              local.get 12
              i32.const 8
              i32.mul
              i32.add
              local.set 16
              local.get 16
              local.get 16
              f64.load
              local.get 25
              local.get 19
              f64.mul
              f64.add
              f64.store
              local.get 31
              local.get 12
              i32.const 8
              i32.mul
              i32.add
              local.set 16
              local.get 16
              local.get 16
              f64.load
              local.get 29
              local.get 19
              f64.mul
              f64.add
              f64.store
              local.get 35
              local.get 12
              i32.const 8
              i32.mul
              i32.add
              local.set 16
              local.get 16
              local.get 16
              f64.load
              local.get 33
              local.get 19
              f64.mul
              f64.add
              f64.store
            end
            i32.const 0
            local.set 13
            block  ;; label = @5
              loop  ;; label = @6
                local.get 13
                i32.const 1
                i32.add
                local.get 9
                i32.ge_u
                br_if 1 (;@5;)
                local.get 15
                local.get 13
                i32.const 8
                i32.mul
                i32.add
                local.set 16
                local.get 16
                v128.load
                local.set 18
                local.get 18
                local.get 22
                local.get 24
                local.get 13
                i32.const 8
                i32.mul
                i32.add
                v128.load
                f64x2.mul
                f64x2.add
                local.set 18
                local.get 18
                local.get 26
                local.get 28
                local.get 13
                i32.const 8
                i32.mul
                i32.add
                v128.load
                f64x2.mul
                f64x2.add
                local.set 18
                local.get 18
                local.get 30
                local.get 32
                local.get 13
                i32.const 8
                i32.mul
                i32.add
                v128.load
                f64x2.mul
                f64x2.add
                local.set 18
                local.get 18
                local.get 34
                local.get 36
                local.get 13
                i32.const 8
                i32.mul
                i32.add
                v128.load
                f64x2.mul
                f64x2.add
                local.set 18
                local.get 16
                local.get 18
                v128.store
                local.get 13
                i32.const 2
                i32.add
                local.set 13
                br 0 (;@6;)
              end
            end
            local.get 13
            local.get 9
            i32.lt_u
            if  ;; label = @5
              local.get 15
              local.get 13
              i32.const 8
              i32.mul
              i32.add
              local.set 16
              local.get 16
              f64.load
              local.set 20
              local.get 20
              local.get 21
              local.get 24
              local.get 13
              i32.const 8
              i32.mul
              i32.add
              f64.load
              f64.mul
              f64.add
              local.set 20
              local.get 20
              local.get 25
              local.get 28
              local.get 13
              i32.const 8
              i32.mul
              i32.add
              f64.load
              f64.mul
              f64.add
              local.set 20
              local.get 20
              local.get 29
              local.get 32
              local.get 13
              i32.const 8
              i32.mul
              i32.add
              f64.load
              f64.mul
              f64.add
              local.set 20
              local.get 20
              local.get 33
              local.get 36
              local.get 13
              i32.const 8
              i32.mul
              i32.add
              f64.load
              f64.mul
              f64.add
              local.set 20
              local.get 16
              local.get 20
              f64.store
            end
            local.get 11
            i32.const 4
            i32.add
            local.set 11
            br 0 (;@4;)
          end
        end
        local.get 11
        local.get 8
        i32.lt_u
        if  ;; label = @3
          local.get 14
          local.get 1
          local.get 10
          local.get 8
          i32.mul
          local.get 11
          i32.add
          i32.const 8
          i32.mul
          i32.add
          local.get 2
          local.get 11
          local.get 9
          i32.mul
          i32.const 8
          i32.mul
          i32.add
          local.get 3
          local.get 11
          local.get 7
          i32.mul
          i32.const 8
          i32.mul
          i32.add
          local.get 4
          local.get 11
          i32.const 8
          i32.mul
          i32.add
          local.get 15
          i32.const 1
          local.get 7
          local.get 8
          local.get 11
          i32.sub
          local.get 9
          call 0
        end
        local.get 10
        i32.const 1
        i32.add
        local.set 10
        br 0 (;@2;)
      end
    end)
  (func (;2;) (type 1) (param i32 i32 i32 i32 i32 i32 i32 i32 f64) (result i32)
    (local i32 i32 i32 v128 v128 f64 i32 i32 i32 i32 v128 v128 v128 v128 v128 v128 v128 v128 v128 v128 v128 v128)
    block  ;; label = @1
      loop  ;; label = @2
        local.get 9
        i32.const 2
        i32.add
        local.get 2
        i32.gt_u
        br_if 1 (;@1;)
        i32.const 0
        local.set 17
        i32.const 0
        local.set 18
        i32.const 0
        local.set 10
        block  ;; label = @3
          loop  ;; label = @4
            local.get 10
            local.get 4
            i32.ge_u
            br_if 1 (;@3;)
            v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
            local.set 23
            v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
            local.set 24
            v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
            local.set 25
            v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
            local.set 26
            v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
            local.set 27
            v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
            local.set 28
            v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
            local.set 29
            v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
            local.set 30
            i32.const 0
            local.set 11
            block  ;; label = @5
              loop  ;; label = @6
                local.get 11
                local.get 3
                i32.ge_u
                br_if 1 (;@5;)
                local.get 1
                local.get 11
                local.get 5
                i32.mul
                local.get 10
                i32.const 0
                i32.add
                i32.add
                i32.const 8
                i32.mul
                i32.add
                v128.load
                local.set 19
                local.get 1
                local.get 11
                local.get 5
                i32.mul
                local.get 10
                i32.const 2
                i32.add
                i32.add
                i32.const 8
                i32.mul
                i32.add
                v128.load
                local.set 20
                local.get 1
                local.get 11
                local.get 5
                i32.mul
                local.get 10
                i32.const 4
                i32.add
                i32.add
                i32.const 8
                i32.mul
                i32.add
                v128.load
                local.set 21
                local.get 1
                local.get 11
                local.get 5
                i32.mul
                local.get 10
                i32.const 6
                i32.add
                i32.add
                i32.const 8
                i32.mul
                i32.add
                v128.load
                local.set 22
                local.get 0
                local.get 9
                i32.const 0
                i32.add
                local.get 3
                i32.mul
                local.get 11
                i32.add
                i32.const 8
                i32.mul
                i32.add
                f64.load
                f64x2.splat
                local.set 12
                local.get 12
                local.get 19
                f64x2.sub
                local.set 13
                local.get 23
                local.get 13
                local.get 13
                f64x2.mul
                f64x2.add
                local.set 23
                local.get 12
                local.get 20
                f64x2.sub
                local.set 13
                local.get 24
                local.get 13
                local.get 13
                f64x2.mul
                f64x2.add
                local.set 24
                local.get 12
                local.get 21
                f64x2.sub
                local.set 13
                local.get 25
                local.get 13
                local.get 13
                f64x2.mul
                f64x2.add
                local.set 25
                local.get 12
                local.get 22
                f64x2.sub
                local.set 13
                local.get 26
                local.get 13
                local.get 13
                f64x2.mul
                f64x2.add
                local.set 26
                local.get 0
                local.get 9
                i32.const 1
                i32.add
                local.get 3
                i32.mul
                local.get 11
                i32.add
                i32.const 8
                i32.mul
                i32.add
                f64.load
                f64x2.splat
                local.set 12
                local.get 12
                local.get 19
                f64x2.sub
                local.set 13
                local.get 27
                local.get 13
                local.get 13
                f64x2.mul
                f64x2.add
                local.set 27
                local.get 12
                local.get 20
                f64x2.sub
                local.set 13
                local.get 28
                local.get 13
                local.get 13
                f64x2.mul
                f64x2.add
                local.set 28
                local.get 12
                local.get 21
                f64x2.sub
                local.set 13
                local.get 29
                local.get 13
                local.get 13
                f64x2.mul
                f64x2.add
                local.set 29
                local.get 12
                local.get 22
                f64x2.sub
                local.set 13
                local.get 30
                local.get 13
                local.get 13
                f64x2.mul
                f64x2.add
                local.set 30
                local.get 11
                i32.const 1
                i32.add
                local.set 11
                br 0 (;@6;)
              end
            end
            local.get 10
            i32.const 0
            i32.add
            local.get 4
            i32.lt_u
            if  ;; label = @5
              local.get 23
              f64x2.extract_lane 0
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 0
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 6
              local.get 9
              i32.const 0
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 0
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              local.get 14
              f64.store
              local.get 27
              f64x2.extract_lane 0
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 0
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 18
                i32.eqz
                local.get 16
                local.get 18
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 18
                end
              end
              local.get 6
              local.get 9
              i32.const 1
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 0
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              local.get 14
              f64.store
            end
            local.get 10
            i32.const 1
            i32.add
            local.get 4
            i32.lt_u
            if  ;; label = @5
              local.get 23
              f64x2.extract_lane 1
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 1
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 6
              local.get 9
              i32.const 0
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 1
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              local.get 14
              f64.store
              local.get 27
              f64x2.extract_lane 1
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 1
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 18
                i32.eqz
                local.get 16
                local.get 18
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 18
                end
              end
              local.get 6
              local.get 9
              i32.const 1
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 1
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              local.get 14
              f64.store
            end
            local.get 10
            i32.const 2
            i32.add
            local.get 4
            i32.lt_u
            if  ;; label = @5
              local.get 24
              f64x2.extract_lane 0
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 2
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 6
              local.get 9
              i32.const 0
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 2
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              local.get 14
              f64.store
              local.get 28
              f64x2.extract_lane 0
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 2
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 18
                i32.eqz
                local.get 16
                local.get 18
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 18
                end
              end
              local.get 6
              local.get 9
              i32.const 1
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 2
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              local.get 14
              f64.store
            end
            local.get 10
            i32.const 3
            i32.add
            local.get 4
            i32.lt_u
            if  ;; label = @5
              local.get 24
              f64x2.extract_lane 1
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 3
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 6
              local.get 9
              i32.const 0
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 3
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              local.get 14
              f64.store
              local.get 28
              f64x2.extract_lane 1
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 3
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 18
                i32.eqz
                local.get 16
                local.get 18
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 18
                end
              end
              local.get 6
              local.get 9
              i32.const 1
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 3
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              local.get 14
              f64.store
            end
            local.get 10
            i32.const 4
            i32.add
            local.get 4
            i32.lt_u
            if  ;; label = @5
              local.get 25
              f64x2.extract_lane 0
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 4
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 6
              local.get 9
              i32.const 0
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 4
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              local.get 14
              f64.store
              local.get 29
              f64x2.extract_lane 0
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 4
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 18
                i32.eqz
                local.get 16
                local.get 18
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 18
                end
              end
              local.get 6
              local.get 9
              i32.const 1
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 4
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              local.get 14
              f64.store
            end
            local.get 10
            i32.const 5
            i32.add
            local.get 4
            i32.lt_u
            if  ;; label = @5
              local.get 25
              f64x2.extract_lane 1
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 5
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 6
              local.get 9
              i32.const 0
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 5
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              local.get 14
              f64.store
              local.get 29
              f64x2.extract_lane 1
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 5
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 18
                i32.eqz
                local.get 16
                local.get 18
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 18
                end
              end
              local.get 6
              local.get 9
              i32.const 1
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 5
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              local.get 14
              f64.store
            end
            local.get 10
            i32.const 6
            i32.add
            local.get 4
            i32.lt_u
            if  ;; label = @5
              local.get 26
              f64x2.extract_lane 0
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 6
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 6
              local.get 9
              i32.const 0
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 6
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              local.get 14
              f64.store
              local.get 30
              f64x2.extract_lane 0
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 6
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 18
                i32.eqz
                local.get 16
                local.get 18
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 18
                end
              end
              local.get 6
              local.get 9
              i32.const 1
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 6
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              local.get 14
              f64.store
            end
            local.get 10
            i32.const 7
            i32.add
            local.get 4
            i32.lt_u
            if  ;; label = @5
              local.get 26
              f64x2.extract_lane 1
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 7
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 6
              local.get 9
              i32.const 0
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 7
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              local.get 14
              f64.store
              local.get 30
              f64x2.extract_lane 1
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 7
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 18
                i32.eqz
                local.get 16
                local.get 18
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 18
                end
              end
              local.get 6
              local.get 9
              i32.const 1
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 7
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              local.get 14
              f64.store
            end
            local.get 10
            i32.const 8
            i32.add
            local.set 10
            br 0 (;@4;)
          end
        end
        local.get 7
        local.get 9
        i32.const 0
        i32.add
        i32.const 4
        i32.mul
        i32.add
        local.get 17
        i32.store
        local.get 7
        local.get 9
        i32.const 1
        i32.add
        i32.const 4
        i32.mul
        i32.add
        local.get 18
        i32.store
        local.get 9
        i32.const 2
        i32.add
        local.set 9
        br 0 (;@2;)
      end
    end
    local.get 9)
  (func (;3;) (type 1) (param i32 i32 i32 i32 i32 i32 i32 i32 f64) (result i32)
    (local i32 i32 i32 v128 v128 f64 i32 i32 i32 v128 v128 v128 v128 v128 v128 v128 v128)
    block  ;; label = @1
      loop  ;; label = @2
        local.get 9
        i32.const 1
        i32.add
        local.get 2
        i32.gt_u
        br_if 1 (;@1;)
        i32.const 0
        local.set 17
        i32.const 0
        local.set 10
        block  ;; label = @3
          loop  ;; label = @4
            local.get 10
            local.get 4
            i32.ge_u
            br_if 1 (;@3;)
            v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
            local.set 22
            v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
            local.set 23
            v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
            local.set 24
            v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
            local.set 25
            i32.const 0
            local.set 11
            block  ;; label = @5
              loop  ;; label = @6
                local.get 11
                local.get 3
                i32.ge_u
                br_if 1 (;@5;)
                local.get 1
                local.get 11
                local.get 5
                i32.mul
                local.get 10
                i32.const 0
                i32.add
                i32.add
                i32.const 8
                i32.mul
                i32.add
                v128.load
                local.set 18
                local.get 1
                local.get 11
                local.get 5
                i32.mul
                local.get 10
                i32.const 2
                i32.add
                i32.add
                i32.const 8
                i32.mul
                i32.add
                v128.load
                local.set 19
                local.get 1
                local.get 11
                local.get 5
                i32.mul
                local.get 10
                i32.const 4
                i32.add
                i32.add
                i32.const 8
                i32.mul
                i32.add
                v128.load
                local.set 20
                local.get 1
                local.get 11
                local.get 5
                i32.mul
                local.get 10
                i32.const 6
                i32.add
                i32.add
                i32.const 8
                i32.mul
                i32.add
                v128.load
                local.set 21
                local.get 0
                local.get 9
                i32.const 0
                i32.add
                local.get 3
                i32.mul
                local.get 11
                i32.add
                i32.const 8
                i32.mul
                i32.add
                f64.load
                f64x2.splat
                local.set 12
                local.get 12
                local.get 18
                f64x2.sub
                local.set 13
                local.get 22
                local.get 13
                local.get 13
                f64x2.mul
                f64x2.add
                local.set 22
                local.get 12
                local.get 19
                f64x2.sub
                local.set 13
                local.get 23
                local.get 13
                local.get 13
                f64x2.mul
                f64x2.add
                local.set 23
                local.get 12
                local.get 20
                f64x2.sub
                local.set 13
                local.get 24
                local.get 13
                local.get 13
                f64x2.mul
                f64x2.add
                local.set 24
                local.get 12
                local.get 21
                f64x2.sub
                local.set 13
                local.get 25
                local.get 13
                local.get 13
                f64x2.mul
                f64x2.add
                local.set 25
                local.get 11
                i32.const 1
                i32.add
                local.set 11
                br 0 (;@6;)
              end
            end
            local.get 10
            i32.const 0
            i32.add
            local.get 4
            i32.lt_u
            if  ;; label = @5
              local.get 22
              f64x2.extract_lane 0
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 0
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 6
              local.get 9
              i32.const 0
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 0
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              local.get 14
              f64.store
            end
            local.get 10
            i32.const 1
            i32.add
            local.get 4
            i32.lt_u
            if  ;; label = @5
              local.get 22
              f64x2.extract_lane 1
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 1
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 6
              local.get 9
              i32.const 0
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 1
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              local.get 14
              f64.store
            end
            local.get 10
            i32.const 2
            i32.add
            local.get 4
            i32.lt_u
            if  ;; label = @5
              local.get 23
              f64x2.extract_lane 0
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 2
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 6
              local.get 9
              i32.const 0
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 2
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              local.get 14
              f64.store
            end
            local.get 10
            i32.const 3
            i32.add
            local.get 4
            i32.lt_u
            if  ;; label = @5
              local.get 23
              f64x2.extract_lane 1
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 3
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 6
              local.get 9
              i32.const 0
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 3
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              local.get 14
              f64.store
            end
            local.get 10
            i32.const 4
            i32.add
            local.get 4
            i32.lt_u
            if  ;; label = @5
              local.get 24
              f64x2.extract_lane 0
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 4
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 6
              local.get 9
              i32.const 0
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 4
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              local.get 14
              f64.store
            end
            local.get 10
            i32.const 5
            i32.add
            local.get 4
            i32.lt_u
            if  ;; label = @5
              local.get 24
              f64x2.extract_lane 1
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 5
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 6
              local.get 9
              i32.const 0
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 5
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              local.get 14
              f64.store
            end
            local.get 10
            i32.const 6
            i32.add
            local.get 4
            i32.lt_u
            if  ;; label = @5
              local.get 25
              f64x2.extract_lane 0
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 6
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 6
              local.get 9
              i32.const 0
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 6
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              local.get 14
              f64.store
            end
            local.get 10
            i32.const 7
            i32.add
            local.get 4
            i32.lt_u
            if  ;; label = @5
              local.get 25
              f64x2.extract_lane 1
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 7
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 6
              local.get 9
              i32.const 0
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 7
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              local.get 14
              f64.store
            end
            local.get 10
            i32.const 8
            i32.add
            local.set 10
            br 0 (;@4;)
          end
        end
        local.get 7
        local.get 9
        i32.const 0
        i32.add
        i32.const 4
        i32.mul
        i32.add
        local.get 17
        i32.store
        local.get 9
        i32.const 1
        i32.add
        local.set 9
        br 0 (;@2;)
      end
    end
    local.get 9)
  (func (;4;) (type 2) (param i32 i32 i32 i32 i32 i32 i32 i32 f64)
    (local i32)
    local.get 0
    local.get 1
    local.get 2
    local.get 3
    local.get 4
    local.get 5
    local.get 6
    local.get 7
    local.get 8
    call 2
    local.set 9
    local.get 0
    local.get 9
    local.get 3
    i32.mul
    i32.const 8
    i32.mul
    i32.add
    local.get 1
    local.get 2
    local.get 9
    i32.sub
    local.get 3
    local.get 4
    local.get 5
    local.get 6
    local.get 9
    local.get 4
    i32.mul
    i32.const 8
    i32.mul
    i32.add
    local.get 7
    local.get 9
    i32.const 4
    i32.mul
    i32.add
    local.get 8
    call 3
    drop)
  (func (;5;) (type 1) (param i32 i32 i32 i32 i32 i32 i32 i32 f64) (result i32)
    (local i32 i32 i32 v128 v128 f64 i32 i32 i32 i32 v128 v128 v128 v128 v128 v128 v128 v128 v128 v128 v128 v128)
    block  ;; label = @1
      loop  ;; label = @2
        local.get 9
        i32.const 2
        i32.add
        local.get 2
        i32.gt_u
        br_if 1 (;@1;)
        local.get 7
        local.get 9
        i32.const 0
        i32.add
        i32.const 4
        i32.mul
        i32.add
        i32.load
        local.set 17
        local.get 7
        local.get 9
        i32.const 1
        i32.add
        i32.const 4
        i32.mul
        i32.add
        i32.load
        local.set 18
        i32.const 0
        local.set 10
        block  ;; label = @3
          loop  ;; label = @4
            local.get 10
            local.get 4
            i32.ge_u
            br_if 1 (;@3;)
            v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
            local.set 23
            v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
            local.set 24
            v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
            local.set 25
            v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
            local.set 26
            v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
            local.set 27
            v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
            local.set 28
            v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
            local.set 29
            v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
            local.set 30
            i32.const 0
            local.set 11
            block  ;; label = @5
              loop  ;; label = @6
                local.get 11
                local.get 3
                i32.ge_u
                br_if 1 (;@5;)
                local.get 1
                local.get 11
                local.get 5
                i32.mul
                local.get 10
                i32.const 0
                i32.add
                i32.add
                i32.const 8
                i32.mul
                i32.add
                v128.load
                local.set 19
                local.get 1
                local.get 11
                local.get 5
                i32.mul
                local.get 10
                i32.const 2
                i32.add
                i32.add
                i32.const 8
                i32.mul
                i32.add
                v128.load
                local.set 20
                local.get 1
                local.get 11
                local.get 5
                i32.mul
                local.get 10
                i32.const 4
                i32.add
                i32.add
                i32.const 8
                i32.mul
                i32.add
                v128.load
                local.set 21
                local.get 1
                local.get 11
                local.get 5
                i32.mul
                local.get 10
                i32.const 6
                i32.add
                i32.add
                i32.const 8
                i32.mul
                i32.add
                v128.load
                local.set 22
                local.get 0
                local.get 9
                i32.const 0
                i32.add
                local.get 3
                i32.mul
                local.get 11
                i32.add
                i32.const 8
                i32.mul
                i32.add
                f64.load
                f64x2.splat
                local.set 12
                local.get 12
                local.get 19
                f64x2.sub
                local.set 13
                local.get 23
                local.get 13
                local.get 13
                f64x2.mul
                f64x2.add
                local.set 23
                local.get 12
                local.get 20
                f64x2.sub
                local.set 13
                local.get 24
                local.get 13
                local.get 13
                f64x2.mul
                f64x2.add
                local.set 24
                local.get 12
                local.get 21
                f64x2.sub
                local.set 13
                local.get 25
                local.get 13
                local.get 13
                f64x2.mul
                f64x2.add
                local.set 25
                local.get 12
                local.get 22
                f64x2.sub
                local.set 13
                local.get 26
                local.get 13
                local.get 13
                f64x2.mul
                f64x2.add
                local.set 26
                local.get 0
                local.get 9
                i32.const 1
                i32.add
                local.get 3
                i32.mul
                local.get 11
                i32.add
                i32.const 8
                i32.mul
                i32.add
                f64.load
                f64x2.splat
                local.set 12
                local.get 12
                local.get 19
                f64x2.sub
                local.set 13
                local.get 27
                local.get 13
                local.get 13
                f64x2.mul
                f64x2.add
                local.set 27
                local.get 12
                local.get 20
                f64x2.sub
                local.set 13
                local.get 28
                local.get 13
                local.get 13
                f64x2.mul
                f64x2.add
                local.set 28
                local.get 12
                local.get 21
                f64x2.sub
                local.set 13
                local.get 29
                local.get 13
                local.get 13
                f64x2.mul
                f64x2.add
                local.set 29
                local.get 12
                local.get 22
                f64x2.sub
                local.set 13
                local.get 30
                local.get 13
                local.get 13
                f64x2.mul
                f64x2.add
                local.set 30
                local.get 11
                i32.const 1
                i32.add
                local.set 11
                br 0 (;@6;)
              end
            end
            local.get 10
            i32.const 0
            i32.add
            local.get 4
            i32.lt_u
            if  ;; label = @5
              local.get 23
              f64x2.extract_lane 0
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 0
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 6
              local.get 9
              i32.const 0
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 0
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              f64.load
              local.get 8
              local.get 14
              f64.mul
              f64.add
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 0
                i32.add
                i32.const 2
                i32.mul
                i32.const 2
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 15
              local.get 14
              f64.store
              local.get 27
              f64x2.extract_lane 0
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 0
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 18
                i32.eqz
                local.get 16
                local.get 18
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 18
                end
              end
              local.get 6
              local.get 9
              i32.const 1
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 0
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              f64.load
              local.get 8
              local.get 14
              f64.mul
              f64.add
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 0
                i32.add
                i32.const 2
                i32.mul
                i32.const 2
                i32.add
                local.set 16
                local.get 18
                i32.eqz
                local.get 16
                local.get 18
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 18
                end
              end
              local.get 15
              local.get 14
              f64.store
            end
            local.get 10
            i32.const 1
            i32.add
            local.get 4
            i32.lt_u
            if  ;; label = @5
              local.get 23
              f64x2.extract_lane 1
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 1
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 6
              local.get 9
              i32.const 0
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 1
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              f64.load
              local.get 8
              local.get 14
              f64.mul
              f64.add
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 1
                i32.add
                i32.const 2
                i32.mul
                i32.const 2
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 15
              local.get 14
              f64.store
              local.get 27
              f64x2.extract_lane 1
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 1
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 18
                i32.eqz
                local.get 16
                local.get 18
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 18
                end
              end
              local.get 6
              local.get 9
              i32.const 1
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 1
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              f64.load
              local.get 8
              local.get 14
              f64.mul
              f64.add
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 1
                i32.add
                i32.const 2
                i32.mul
                i32.const 2
                i32.add
                local.set 16
                local.get 18
                i32.eqz
                local.get 16
                local.get 18
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 18
                end
              end
              local.get 15
              local.get 14
              f64.store
            end
            local.get 10
            i32.const 2
            i32.add
            local.get 4
            i32.lt_u
            if  ;; label = @5
              local.get 24
              f64x2.extract_lane 0
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 2
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 6
              local.get 9
              i32.const 0
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 2
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              f64.load
              local.get 8
              local.get 14
              f64.mul
              f64.add
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 2
                i32.add
                i32.const 2
                i32.mul
                i32.const 2
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 15
              local.get 14
              f64.store
              local.get 28
              f64x2.extract_lane 0
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 2
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 18
                i32.eqz
                local.get 16
                local.get 18
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 18
                end
              end
              local.get 6
              local.get 9
              i32.const 1
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 2
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              f64.load
              local.get 8
              local.get 14
              f64.mul
              f64.add
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 2
                i32.add
                i32.const 2
                i32.mul
                i32.const 2
                i32.add
                local.set 16
                local.get 18
                i32.eqz
                local.get 16
                local.get 18
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 18
                end
              end
              local.get 15
              local.get 14
              f64.store
            end
            local.get 10
            i32.const 3
            i32.add
            local.get 4
            i32.lt_u
            if  ;; label = @5
              local.get 24
              f64x2.extract_lane 1
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 3
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 6
              local.get 9
              i32.const 0
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 3
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              f64.load
              local.get 8
              local.get 14
              f64.mul
              f64.add
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 3
                i32.add
                i32.const 2
                i32.mul
                i32.const 2
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 15
              local.get 14
              f64.store
              local.get 28
              f64x2.extract_lane 1
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 3
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 18
                i32.eqz
                local.get 16
                local.get 18
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 18
                end
              end
              local.get 6
              local.get 9
              i32.const 1
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 3
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              f64.load
              local.get 8
              local.get 14
              f64.mul
              f64.add
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 3
                i32.add
                i32.const 2
                i32.mul
                i32.const 2
                i32.add
                local.set 16
                local.get 18
                i32.eqz
                local.get 16
                local.get 18
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 18
                end
              end
              local.get 15
              local.get 14
              f64.store
            end
            local.get 10
            i32.const 4
            i32.add
            local.get 4
            i32.lt_u
            if  ;; label = @5
              local.get 25
              f64x2.extract_lane 0
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 4
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 6
              local.get 9
              i32.const 0
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 4
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              f64.load
              local.get 8
              local.get 14
              f64.mul
              f64.add
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 4
                i32.add
                i32.const 2
                i32.mul
                i32.const 2
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 15
              local.get 14
              f64.store
              local.get 29
              f64x2.extract_lane 0
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 4
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 18
                i32.eqz
                local.get 16
                local.get 18
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 18
                end
              end
              local.get 6
              local.get 9
              i32.const 1
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 4
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              f64.load
              local.get 8
              local.get 14
              f64.mul
              f64.add
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 4
                i32.add
                i32.const 2
                i32.mul
                i32.const 2
                i32.add
                local.set 16
                local.get 18
                i32.eqz
                local.get 16
                local.get 18
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 18
                end
              end
              local.get 15
              local.get 14
              f64.store
            end
            local.get 10
            i32.const 5
            i32.add
            local.get 4
            i32.lt_u
            if  ;; label = @5
              local.get 25
              f64x2.extract_lane 1
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 5
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 6
              local.get 9
              i32.const 0
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 5
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              f64.load
              local.get 8
              local.get 14
              f64.mul
              f64.add
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 5
                i32.add
                i32.const 2
                i32.mul
                i32.const 2
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 15
              local.get 14
              f64.store
              local.get 29
              f64x2.extract_lane 1
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 5
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 18
                i32.eqz
                local.get 16
                local.get 18
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 18
                end
              end
              local.get 6
              local.get 9
              i32.const 1
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 5
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              f64.load
              local.get 8
              local.get 14
              f64.mul
              f64.add
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 5
                i32.add
                i32.const 2
                i32.mul
                i32.const 2
                i32.add
                local.set 16
                local.get 18
                i32.eqz
                local.get 16
                local.get 18
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 18
                end
              end
              local.get 15
              local.get 14
              f64.store
            end
            local.get 10
            i32.const 6
            i32.add
            local.get 4
            i32.lt_u
            if  ;; label = @5
              local.get 26
              f64x2.extract_lane 0
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 6
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 6
              local.get 9
              i32.const 0
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 6
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              f64.load
              local.get 8
              local.get 14
              f64.mul
              f64.add
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 6
                i32.add
                i32.const 2
                i32.mul
                i32.const 2
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 15
              local.get 14
              f64.store
              local.get 30
              f64x2.extract_lane 0
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 6
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 18
                i32.eqz
                local.get 16
                local.get 18
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 18
                end
              end
              local.get 6
              local.get 9
              i32.const 1
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 6
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              f64.load
              local.get 8
              local.get 14
              f64.mul
              f64.add
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 6
                i32.add
                i32.const 2
                i32.mul
                i32.const 2
                i32.add
                local.set 16
                local.get 18
                i32.eqz
                local.get 16
                local.get 18
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 18
                end
              end
              local.get 15
              local.get 14
              f64.store
            end
            local.get 10
            i32.const 7
            i32.add
            local.get 4
            i32.lt_u
            if  ;; label = @5
              local.get 26
              f64x2.extract_lane 1
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 7
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 6
              local.get 9
              i32.const 0
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 7
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              f64.load
              local.get 8
              local.get 14
              f64.mul
              f64.add
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 7
                i32.add
                i32.const 2
                i32.mul
                i32.const 2
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 15
              local.get 14
              f64.store
              local.get 30
              f64x2.extract_lane 1
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 7
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 18
                i32.eqz
                local.get 16
                local.get 18
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 18
                end
              end
              local.get 6
              local.get 9
              i32.const 1
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 7
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              f64.load
              local.get 8
              local.get 14
              f64.mul
              f64.add
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 7
                i32.add
                i32.const 2
                i32.mul
                i32.const 2
                i32.add
                local.set 16
                local.get 18
                i32.eqz
                local.get 16
                local.get 18
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 18
                end
              end
              local.get 15
              local.get 14
              f64.store
            end
            local.get 10
            i32.const 8
            i32.add
            local.set 10
            br 0 (;@4;)
          end
        end
        local.get 7
        local.get 9
        i32.const 0
        i32.add
        i32.const 4
        i32.mul
        i32.add
        local.get 17
        i32.store
        local.get 7
        local.get 9
        i32.const 1
        i32.add
        i32.const 4
        i32.mul
        i32.add
        local.get 18
        i32.store
        local.get 9
        i32.const 2
        i32.add
        local.set 9
        br 0 (;@2;)
      end
    end
    local.get 9)
  (func (;6;) (type 1) (param i32 i32 i32 i32 i32 i32 i32 i32 f64) (result i32)
    (local i32 i32 i32 v128 v128 f64 i32 i32 i32 v128 v128 v128 v128 v128 v128 v128 v128)
    block  ;; label = @1
      loop  ;; label = @2
        local.get 9
        i32.const 1
        i32.add
        local.get 2
        i32.gt_u
        br_if 1 (;@1;)
        local.get 7
        local.get 9
        i32.const 0
        i32.add
        i32.const 4
        i32.mul
        i32.add
        i32.load
        local.set 17
        i32.const 0
        local.set 10
        block  ;; label = @3
          loop  ;; label = @4
            local.get 10
            local.get 4
            i32.ge_u
            br_if 1 (;@3;)
            v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
            local.set 22
            v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
            local.set 23
            v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
            local.set 24
            v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
            local.set 25
            i32.const 0
            local.set 11
            block  ;; label = @5
              loop  ;; label = @6
                local.get 11
                local.get 3
                i32.ge_u
                br_if 1 (;@5;)
                local.get 1
                local.get 11
                local.get 5
                i32.mul
                local.get 10
                i32.const 0
                i32.add
                i32.add
                i32.const 8
                i32.mul
                i32.add
                v128.load
                local.set 18
                local.get 1
                local.get 11
                local.get 5
                i32.mul
                local.get 10
                i32.const 2
                i32.add
                i32.add
                i32.const 8
                i32.mul
                i32.add
                v128.load
                local.set 19
                local.get 1
                local.get 11
                local.get 5
                i32.mul
                local.get 10
                i32.const 4
                i32.add
                i32.add
                i32.const 8
                i32.mul
                i32.add
                v128.load
                local.set 20
                local.get 1
                local.get 11
                local.get 5
                i32.mul
                local.get 10
                i32.const 6
                i32.add
                i32.add
                i32.const 8
                i32.mul
                i32.add
                v128.load
                local.set 21
                local.get 0
                local.get 9
                i32.const 0
                i32.add
                local.get 3
                i32.mul
                local.get 11
                i32.add
                i32.const 8
                i32.mul
                i32.add
                f64.load
                f64x2.splat
                local.set 12
                local.get 12
                local.get 18
                f64x2.sub
                local.set 13
                local.get 22
                local.get 13
                local.get 13
                f64x2.mul
                f64x2.add
                local.set 22
                local.get 12
                local.get 19
                f64x2.sub
                local.set 13
                local.get 23
                local.get 13
                local.get 13
                f64x2.mul
                f64x2.add
                local.set 23
                local.get 12
                local.get 20
                f64x2.sub
                local.set 13
                local.get 24
                local.get 13
                local.get 13
                f64x2.mul
                f64x2.add
                local.set 24
                local.get 12
                local.get 21
                f64x2.sub
                local.set 13
                local.get 25
                local.get 13
                local.get 13
                f64x2.mul
                f64x2.add
                local.set 25
                local.get 11
                i32.const 1
                i32.add
                local.set 11
                br 0 (;@6;)
              end
            end
            local.get 10
            i32.const 0
            i32.add
            local.get 4
            i32.lt_u
            if  ;; label = @5
              local.get 22
              f64x2.extract_lane 0
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 0
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 6
              local.get 9
              i32.const 0
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 0
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              f64.load
              local.get 8
              local.get 14
              f64.mul
              f64.add
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 0
                i32.add
                i32.const 2
                i32.mul
                i32.const 2
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 15
              local.get 14
              f64.store
            end
            local.get 10
            i32.const 1
            i32.add
            local.get 4
            i32.lt_u
            if  ;; label = @5
              local.get 22
              f64x2.extract_lane 1
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 1
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 6
              local.get 9
              i32.const 0
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 1
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              f64.load
              local.get 8
              local.get 14
              f64.mul
              f64.add
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 1
                i32.add
                i32.const 2
                i32.mul
                i32.const 2
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 15
              local.get 14
              f64.store
            end
            local.get 10
            i32.const 2
            i32.add
            local.get 4
            i32.lt_u
            if  ;; label = @5
              local.get 23
              f64x2.extract_lane 0
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 2
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 6
              local.get 9
              i32.const 0
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 2
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              f64.load
              local.get 8
              local.get 14
              f64.mul
              f64.add
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 2
                i32.add
                i32.const 2
                i32.mul
                i32.const 2
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 15
              local.get 14
              f64.store
            end
            local.get 10
            i32.const 3
            i32.add
            local.get 4
            i32.lt_u
            if  ;; label = @5
              local.get 23
              f64x2.extract_lane 1
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 3
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 6
              local.get 9
              i32.const 0
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 3
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              f64.load
              local.get 8
              local.get 14
              f64.mul
              f64.add
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 3
                i32.add
                i32.const 2
                i32.mul
                i32.const 2
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 15
              local.get 14
              f64.store
            end
            local.get 10
            i32.const 4
            i32.add
            local.get 4
            i32.lt_u
            if  ;; label = @5
              local.get 24
              f64x2.extract_lane 0
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 4
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 6
              local.get 9
              i32.const 0
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 4
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              f64.load
              local.get 8
              local.get 14
              f64.mul
              f64.add
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 4
                i32.add
                i32.const 2
                i32.mul
                i32.const 2
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 15
              local.get 14
              f64.store
            end
            local.get 10
            i32.const 5
            i32.add
            local.get 4
            i32.lt_u
            if  ;; label = @5
              local.get 24
              f64x2.extract_lane 1
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 5
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 6
              local.get 9
              i32.const 0
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 5
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              f64.load
              local.get 8
              local.get 14
              f64.mul
              f64.add
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 5
                i32.add
                i32.const 2
                i32.mul
                i32.const 2
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 15
              local.get 14
              f64.store
            end
            local.get 10
            i32.const 6
            i32.add
            local.get 4
            i32.lt_u
            if  ;; label = @5
              local.get 25
              f64x2.extract_lane 0
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 6
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 6
              local.get 9
              i32.const 0
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 6
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              f64.load
              local.get 8
              local.get 14
              f64.mul
              f64.add
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 6
                i32.add
                i32.const 2
                i32.mul
                i32.const 2
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 15
              local.get 14
              f64.store
            end
            local.get 10
            i32.const 7
            i32.add
            local.get 4
            i32.lt_u
            if  ;; label = @5
              local.get 25
              f64x2.extract_lane 1
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 7
                i32.add
                i32.const 2
                i32.mul
                i32.const 1
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 6
              local.get 9
              i32.const 0
              i32.add
              local.get 4
              i32.mul
              local.get 10
              i32.const 7
              i32.add
              i32.add
              i32.const 8
              i32.mul
              i32.add
              local.set 15
              local.get 15
              f64.load
              local.get 8
              local.get 14
              f64.mul
              f64.add
              local.set 14
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.le
              i32.eqz
              if  ;; label = @6
                local.get 10
                i32.const 7
                i32.add
                i32.const 2
                i32.mul
                i32.const 2
                i32.add
                local.set 16
                local.get 17
                i32.eqz
                local.get 16
                local.get 17
                i32.lt_u
                i32.or
                if  ;; label = @7
                  local.get 16
                  local.set 17
                end
              end
              local.get 15
              local.get 14
              f64.store
            end
            local.get 10
            i32.const 8
            i32.add
            local.set 10
            br 0 (;@4;)
          end
        end
        local.get 7
        local.get 9
        i32.const 0
        i32.add
        i32.const 4
        i32.mul
        i32.add
        local.get 17
        i32.store
        local.get 9
        i32.const 1
        i32.add
        local.set 9
        br 0 (;@2;)
      end
    end
    local.get 9)
  (func (;7;) (type 2) (param i32 i32 i32 i32 i32 i32 i32 i32 f64)
    (local i32)
    local.get 0
    local.get 1
    local.get 2
    local.get 3
    local.get 4
    local.get 5
    local.get 6
    local.get 7
    local.get 8
    call 5
    local.set 9
    local.get 0
    local.get 9
    local.get 3
    i32.mul
    i32.const 8
    i32.mul
    i32.add
    local.get 1
    local.get 2
    local.get 9
    i32.sub
    local.get 3
    local.get 4
    local.get 5
    local.get 6
    local.get 9
    local.get 4
    i32.mul
    i32.const 8
    i32.mul
    i32.add
    local.get 7
    local.get 9
    i32.const 4
    i32.mul
    i32.add
    local.get 8
    call 6
    drop)
  (memory (;0;) 1)
  (export "memory" (memory 0))
  (export "prototype" (func 1))
  (export "feature" (func 4))
  (export "grid" (func 7)))

