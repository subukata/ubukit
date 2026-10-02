;; RECONSTRUCTED FROM BINARY, not recovered original source.
;; Input SHA-256: 7ebf62044d0c023db3a9c090bbe7e3abcb1186ef86acb30ae688a14df4abba20
;; Tool: wabt 1.0.39; readWasm(readDebugNames:true,simd:true), applyNames, toText.
;; Original comments, source formatting, authorship and license are NOT recovered.
(module
  (type (;0;) (func (param i32 i32 i32 i32 i32 i32 i32 i32 i32) (result i32)))
  (type (;1;) (func (param i32 i32 i32 i32 i32 i32 i32 i32 i32)))
  (func (;0;) (type 0) (param i32 i32 i32 i32 i32 i32 i32 i32 i32) (result i32)
    (local i32 i32 i32 v128 v128 f64 i32 i32 f64 i32 i32 f64 v128 v128 v128 v128 v128 v128 v128 v128 v128 v128 v128 v128)
    block  ;; label = @1
      loop  ;; label = @2
        local.get 9
        i32.const 2
        i32.add
        local.get 2
        i32.gt_u
        br_if 1 (;@1;)
        f64.const inf (;=inf;)
        local.set 17
        i32.const 0
        local.set 15
        i32.const 0
        local.set 16
        f64.const inf (;=inf;)
        local.set 20
        i32.const 0
        local.set 18
        i32.const 0
        local.set 19
        i32.const 0
        local.set 10
        block  ;; label = @3
          loop  ;; label = @4
            local.get 10
            local.get 4
            i32.ge_u
            br_if 1 (;@3;)
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
            v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
            local.set 31
            v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
            local.set 32
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
                local.set 21
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
                local.set 22
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
                local.set 23
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
                local.set 24
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
                local.get 12
                local.get 23
                f64x2.sub
                local.set 13
                local.get 27
                local.get 13
                local.get 13
                f64x2.mul
                f64x2.add
                local.set 27
                local.get 12
                local.get 24
                f64x2.sub
                local.set 13
                local.get 28
                local.get 13
                local.get 13
                f64x2.mul
                f64x2.add
                local.set 28
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
                local.get 12
                local.get 23
                f64x2.sub
                local.set 13
                local.get 31
                local.get 13
                local.get 13
                f64x2.mul
                f64x2.add
                local.set 31
                local.get 12
                local.get 24
                f64x2.sub
                local.set 13
                local.get 32
                local.get 13
                local.get 13
                f64x2.mul
                f64x2.add
                local.set 32
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
              local.get 25
              f64x2.extract_lane 0
              local.set 14
              local.get 14
              local.get 14
              f64.ne
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.gt
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.gt
              local.get 14
              f64.const 0x1p-1022 (;=2.22507e-308;)
              f64.lt
              i32.and
              i32.or
              i32.or
              if  ;; label = @6
                i32.const 1
                local.set 16
              end
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.eq
              if  ;; label = @6
                i32.const 0
                local.set 11
                block  ;; label = @7
                  loop  ;; label = @8
                    local.get 11
                    local.get 3
                    i32.ge_u
                    br_if 1 (;@7;)
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
                    f64.load
                    f64.ne
                    if  ;; label = @9
                      i32.const 1
                      local.set 16
                      br 2 (;@7;)
                    end
                    local.get 11
                    i32.const 1
                    i32.add
                    local.set 11
                    br 0 (;@8;)
                  end
                end
              end
              local.get 14
              local.get 17
              f64.lt
              if  ;; label = @6
                local.get 14
                local.set 17
                local.get 10
                i32.const 0
                i32.add
                local.set 15
              end
              local.get 29
              f64x2.extract_lane 0
              local.set 14
              local.get 14
              local.get 14
              f64.ne
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.gt
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.gt
              local.get 14
              f64.const 0x1p-1022 (;=2.22507e-308;)
              f64.lt
              i32.and
              i32.or
              i32.or
              if  ;; label = @6
                i32.const 1
                local.set 19
              end
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.eq
              if  ;; label = @6
                i32.const 0
                local.set 11
                block  ;; label = @7
                  loop  ;; label = @8
                    local.get 11
                    local.get 3
                    i32.ge_u
                    br_if 1 (;@7;)
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
                    f64.load
                    f64.ne
                    if  ;; label = @9
                      i32.const 1
                      local.set 19
                      br 2 (;@7;)
                    end
                    local.get 11
                    i32.const 1
                    i32.add
                    local.set 11
                    br 0 (;@8;)
                  end
                end
              end
              local.get 14
              local.get 20
              f64.lt
              if  ;; label = @6
                local.get 14
                local.set 20
                local.get 10
                i32.const 0
                i32.add
                local.set 18
              end
            end
            local.get 10
            i32.const 1
            i32.add
            local.get 4
            i32.lt_u
            if  ;; label = @5
              local.get 25
              f64x2.extract_lane 1
              local.set 14
              local.get 14
              local.get 14
              f64.ne
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.gt
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.gt
              local.get 14
              f64.const 0x1p-1022 (;=2.22507e-308;)
              f64.lt
              i32.and
              i32.or
              i32.or
              if  ;; label = @6
                i32.const 1
                local.set 16
              end
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.eq
              if  ;; label = @6
                i32.const 0
                local.set 11
                block  ;; label = @7
                  loop  ;; label = @8
                    local.get 11
                    local.get 3
                    i32.ge_u
                    br_if 1 (;@7;)
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
                    local.get 1
                    local.get 11
                    local.get 5
                    i32.mul
                    local.get 10
                    i32.const 1
                    i32.add
                    i32.add
                    i32.const 8
                    i32.mul
                    i32.add
                    f64.load
                    f64.ne
                    if  ;; label = @9
                      i32.const 1
                      local.set 16
                      br 2 (;@7;)
                    end
                    local.get 11
                    i32.const 1
                    i32.add
                    local.set 11
                    br 0 (;@8;)
                  end
                end
              end
              local.get 14
              local.get 17
              f64.lt
              if  ;; label = @6
                local.get 14
                local.set 17
                local.get 10
                i32.const 1
                i32.add
                local.set 15
              end
              local.get 29
              f64x2.extract_lane 1
              local.set 14
              local.get 14
              local.get 14
              f64.ne
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.gt
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.gt
              local.get 14
              f64.const 0x1p-1022 (;=2.22507e-308;)
              f64.lt
              i32.and
              i32.or
              i32.or
              if  ;; label = @6
                i32.const 1
                local.set 19
              end
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.eq
              if  ;; label = @6
                i32.const 0
                local.set 11
                block  ;; label = @7
                  loop  ;; label = @8
                    local.get 11
                    local.get 3
                    i32.ge_u
                    br_if 1 (;@7;)
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
                    local.get 1
                    local.get 11
                    local.get 5
                    i32.mul
                    local.get 10
                    i32.const 1
                    i32.add
                    i32.add
                    i32.const 8
                    i32.mul
                    i32.add
                    f64.load
                    f64.ne
                    if  ;; label = @9
                      i32.const 1
                      local.set 19
                      br 2 (;@7;)
                    end
                    local.get 11
                    i32.const 1
                    i32.add
                    local.set 11
                    br 0 (;@8;)
                  end
                end
              end
              local.get 14
              local.get 20
              f64.lt
              if  ;; label = @6
                local.get 14
                local.set 20
                local.get 10
                i32.const 1
                i32.add
                local.set 18
              end
            end
            local.get 10
            i32.const 2
            i32.add
            local.get 4
            i32.lt_u
            if  ;; label = @5
              local.get 26
              f64x2.extract_lane 0
              local.set 14
              local.get 14
              local.get 14
              f64.ne
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.gt
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.gt
              local.get 14
              f64.const 0x1p-1022 (;=2.22507e-308;)
              f64.lt
              i32.and
              i32.or
              i32.or
              if  ;; label = @6
                i32.const 1
                local.set 16
              end
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.eq
              if  ;; label = @6
                i32.const 0
                local.set 11
                block  ;; label = @7
                  loop  ;; label = @8
                    local.get 11
                    local.get 3
                    i32.ge_u
                    br_if 1 (;@7;)
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
                    f64.load
                    f64.ne
                    if  ;; label = @9
                      i32.const 1
                      local.set 16
                      br 2 (;@7;)
                    end
                    local.get 11
                    i32.const 1
                    i32.add
                    local.set 11
                    br 0 (;@8;)
                  end
                end
              end
              local.get 14
              local.get 17
              f64.lt
              if  ;; label = @6
                local.get 14
                local.set 17
                local.get 10
                i32.const 2
                i32.add
                local.set 15
              end
              local.get 30
              f64x2.extract_lane 0
              local.set 14
              local.get 14
              local.get 14
              f64.ne
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.gt
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.gt
              local.get 14
              f64.const 0x1p-1022 (;=2.22507e-308;)
              f64.lt
              i32.and
              i32.or
              i32.or
              if  ;; label = @6
                i32.const 1
                local.set 19
              end
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.eq
              if  ;; label = @6
                i32.const 0
                local.set 11
                block  ;; label = @7
                  loop  ;; label = @8
                    local.get 11
                    local.get 3
                    i32.ge_u
                    br_if 1 (;@7;)
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
                    f64.load
                    f64.ne
                    if  ;; label = @9
                      i32.const 1
                      local.set 19
                      br 2 (;@7;)
                    end
                    local.get 11
                    i32.const 1
                    i32.add
                    local.set 11
                    br 0 (;@8;)
                  end
                end
              end
              local.get 14
              local.get 20
              f64.lt
              if  ;; label = @6
                local.get 14
                local.set 20
                local.get 10
                i32.const 2
                i32.add
                local.set 18
              end
            end
            local.get 10
            i32.const 3
            i32.add
            local.get 4
            i32.lt_u
            if  ;; label = @5
              local.get 26
              f64x2.extract_lane 1
              local.set 14
              local.get 14
              local.get 14
              f64.ne
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.gt
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.gt
              local.get 14
              f64.const 0x1p-1022 (;=2.22507e-308;)
              f64.lt
              i32.and
              i32.or
              i32.or
              if  ;; label = @6
                i32.const 1
                local.set 16
              end
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.eq
              if  ;; label = @6
                i32.const 0
                local.set 11
                block  ;; label = @7
                  loop  ;; label = @8
                    local.get 11
                    local.get 3
                    i32.ge_u
                    br_if 1 (;@7;)
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
                    local.get 1
                    local.get 11
                    local.get 5
                    i32.mul
                    local.get 10
                    i32.const 3
                    i32.add
                    i32.add
                    i32.const 8
                    i32.mul
                    i32.add
                    f64.load
                    f64.ne
                    if  ;; label = @9
                      i32.const 1
                      local.set 16
                      br 2 (;@7;)
                    end
                    local.get 11
                    i32.const 1
                    i32.add
                    local.set 11
                    br 0 (;@8;)
                  end
                end
              end
              local.get 14
              local.get 17
              f64.lt
              if  ;; label = @6
                local.get 14
                local.set 17
                local.get 10
                i32.const 3
                i32.add
                local.set 15
              end
              local.get 30
              f64x2.extract_lane 1
              local.set 14
              local.get 14
              local.get 14
              f64.ne
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.gt
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.gt
              local.get 14
              f64.const 0x1p-1022 (;=2.22507e-308;)
              f64.lt
              i32.and
              i32.or
              i32.or
              if  ;; label = @6
                i32.const 1
                local.set 19
              end
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.eq
              if  ;; label = @6
                i32.const 0
                local.set 11
                block  ;; label = @7
                  loop  ;; label = @8
                    local.get 11
                    local.get 3
                    i32.ge_u
                    br_if 1 (;@7;)
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
                    local.get 1
                    local.get 11
                    local.get 5
                    i32.mul
                    local.get 10
                    i32.const 3
                    i32.add
                    i32.add
                    i32.const 8
                    i32.mul
                    i32.add
                    f64.load
                    f64.ne
                    if  ;; label = @9
                      i32.const 1
                      local.set 19
                      br 2 (;@7;)
                    end
                    local.get 11
                    i32.const 1
                    i32.add
                    local.set 11
                    br 0 (;@8;)
                  end
                end
              end
              local.get 14
              local.get 20
              f64.lt
              if  ;; label = @6
                local.get 14
                local.set 20
                local.get 10
                i32.const 3
                i32.add
                local.set 18
              end
            end
            local.get 10
            i32.const 4
            i32.add
            local.get 4
            i32.lt_u
            if  ;; label = @5
              local.get 27
              f64x2.extract_lane 0
              local.set 14
              local.get 14
              local.get 14
              f64.ne
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.gt
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.gt
              local.get 14
              f64.const 0x1p-1022 (;=2.22507e-308;)
              f64.lt
              i32.and
              i32.or
              i32.or
              if  ;; label = @6
                i32.const 1
                local.set 16
              end
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.eq
              if  ;; label = @6
                i32.const 0
                local.set 11
                block  ;; label = @7
                  loop  ;; label = @8
                    local.get 11
                    local.get 3
                    i32.ge_u
                    br_if 1 (;@7;)
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
                    f64.load
                    f64.ne
                    if  ;; label = @9
                      i32.const 1
                      local.set 16
                      br 2 (;@7;)
                    end
                    local.get 11
                    i32.const 1
                    i32.add
                    local.set 11
                    br 0 (;@8;)
                  end
                end
              end
              local.get 14
              local.get 17
              f64.lt
              if  ;; label = @6
                local.get 14
                local.set 17
                local.get 10
                i32.const 4
                i32.add
                local.set 15
              end
              local.get 31
              f64x2.extract_lane 0
              local.set 14
              local.get 14
              local.get 14
              f64.ne
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.gt
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.gt
              local.get 14
              f64.const 0x1p-1022 (;=2.22507e-308;)
              f64.lt
              i32.and
              i32.or
              i32.or
              if  ;; label = @6
                i32.const 1
                local.set 19
              end
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.eq
              if  ;; label = @6
                i32.const 0
                local.set 11
                block  ;; label = @7
                  loop  ;; label = @8
                    local.get 11
                    local.get 3
                    i32.ge_u
                    br_if 1 (;@7;)
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
                    f64.load
                    f64.ne
                    if  ;; label = @9
                      i32.const 1
                      local.set 19
                      br 2 (;@7;)
                    end
                    local.get 11
                    i32.const 1
                    i32.add
                    local.set 11
                    br 0 (;@8;)
                  end
                end
              end
              local.get 14
              local.get 20
              f64.lt
              if  ;; label = @6
                local.get 14
                local.set 20
                local.get 10
                i32.const 4
                i32.add
                local.set 18
              end
            end
            local.get 10
            i32.const 5
            i32.add
            local.get 4
            i32.lt_u
            if  ;; label = @5
              local.get 27
              f64x2.extract_lane 1
              local.set 14
              local.get 14
              local.get 14
              f64.ne
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.gt
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.gt
              local.get 14
              f64.const 0x1p-1022 (;=2.22507e-308;)
              f64.lt
              i32.and
              i32.or
              i32.or
              if  ;; label = @6
                i32.const 1
                local.set 16
              end
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.eq
              if  ;; label = @6
                i32.const 0
                local.set 11
                block  ;; label = @7
                  loop  ;; label = @8
                    local.get 11
                    local.get 3
                    i32.ge_u
                    br_if 1 (;@7;)
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
                    local.get 1
                    local.get 11
                    local.get 5
                    i32.mul
                    local.get 10
                    i32.const 5
                    i32.add
                    i32.add
                    i32.const 8
                    i32.mul
                    i32.add
                    f64.load
                    f64.ne
                    if  ;; label = @9
                      i32.const 1
                      local.set 16
                      br 2 (;@7;)
                    end
                    local.get 11
                    i32.const 1
                    i32.add
                    local.set 11
                    br 0 (;@8;)
                  end
                end
              end
              local.get 14
              local.get 17
              f64.lt
              if  ;; label = @6
                local.get 14
                local.set 17
                local.get 10
                i32.const 5
                i32.add
                local.set 15
              end
              local.get 31
              f64x2.extract_lane 1
              local.set 14
              local.get 14
              local.get 14
              f64.ne
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.gt
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.gt
              local.get 14
              f64.const 0x1p-1022 (;=2.22507e-308;)
              f64.lt
              i32.and
              i32.or
              i32.or
              if  ;; label = @6
                i32.const 1
                local.set 19
              end
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.eq
              if  ;; label = @6
                i32.const 0
                local.set 11
                block  ;; label = @7
                  loop  ;; label = @8
                    local.get 11
                    local.get 3
                    i32.ge_u
                    br_if 1 (;@7;)
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
                    local.get 1
                    local.get 11
                    local.get 5
                    i32.mul
                    local.get 10
                    i32.const 5
                    i32.add
                    i32.add
                    i32.const 8
                    i32.mul
                    i32.add
                    f64.load
                    f64.ne
                    if  ;; label = @9
                      i32.const 1
                      local.set 19
                      br 2 (;@7;)
                    end
                    local.get 11
                    i32.const 1
                    i32.add
                    local.set 11
                    br 0 (;@8;)
                  end
                end
              end
              local.get 14
              local.get 20
              f64.lt
              if  ;; label = @6
                local.get 14
                local.set 20
                local.get 10
                i32.const 5
                i32.add
                local.set 18
              end
            end
            local.get 10
            i32.const 6
            i32.add
            local.get 4
            i32.lt_u
            if  ;; label = @5
              local.get 28
              f64x2.extract_lane 0
              local.set 14
              local.get 14
              local.get 14
              f64.ne
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.gt
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.gt
              local.get 14
              f64.const 0x1p-1022 (;=2.22507e-308;)
              f64.lt
              i32.and
              i32.or
              i32.or
              if  ;; label = @6
                i32.const 1
                local.set 16
              end
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.eq
              if  ;; label = @6
                i32.const 0
                local.set 11
                block  ;; label = @7
                  loop  ;; label = @8
                    local.get 11
                    local.get 3
                    i32.ge_u
                    br_if 1 (;@7;)
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
                    f64.load
                    f64.ne
                    if  ;; label = @9
                      i32.const 1
                      local.set 16
                      br 2 (;@7;)
                    end
                    local.get 11
                    i32.const 1
                    i32.add
                    local.set 11
                    br 0 (;@8;)
                  end
                end
              end
              local.get 14
              local.get 17
              f64.lt
              if  ;; label = @6
                local.get 14
                local.set 17
                local.get 10
                i32.const 6
                i32.add
                local.set 15
              end
              local.get 32
              f64x2.extract_lane 0
              local.set 14
              local.get 14
              local.get 14
              f64.ne
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.gt
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.gt
              local.get 14
              f64.const 0x1p-1022 (;=2.22507e-308;)
              f64.lt
              i32.and
              i32.or
              i32.or
              if  ;; label = @6
                i32.const 1
                local.set 19
              end
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.eq
              if  ;; label = @6
                i32.const 0
                local.set 11
                block  ;; label = @7
                  loop  ;; label = @8
                    local.get 11
                    local.get 3
                    i32.ge_u
                    br_if 1 (;@7;)
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
                    f64.load
                    f64.ne
                    if  ;; label = @9
                      i32.const 1
                      local.set 19
                      br 2 (;@7;)
                    end
                    local.get 11
                    i32.const 1
                    i32.add
                    local.set 11
                    br 0 (;@8;)
                  end
                end
              end
              local.get 14
              local.get 20
              f64.lt
              if  ;; label = @6
                local.get 14
                local.set 20
                local.get 10
                i32.const 6
                i32.add
                local.set 18
              end
            end
            local.get 10
            i32.const 7
            i32.add
            local.get 4
            i32.lt_u
            if  ;; label = @5
              local.get 28
              f64x2.extract_lane 1
              local.set 14
              local.get 14
              local.get 14
              f64.ne
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.gt
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.gt
              local.get 14
              f64.const 0x1p-1022 (;=2.22507e-308;)
              f64.lt
              i32.and
              i32.or
              i32.or
              if  ;; label = @6
                i32.const 1
                local.set 16
              end
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.eq
              if  ;; label = @6
                i32.const 0
                local.set 11
                block  ;; label = @7
                  loop  ;; label = @8
                    local.get 11
                    local.get 3
                    i32.ge_u
                    br_if 1 (;@7;)
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
                    local.get 1
                    local.get 11
                    local.get 5
                    i32.mul
                    local.get 10
                    i32.const 7
                    i32.add
                    i32.add
                    i32.const 8
                    i32.mul
                    i32.add
                    f64.load
                    f64.ne
                    if  ;; label = @9
                      i32.const 1
                      local.set 16
                      br 2 (;@7;)
                    end
                    local.get 11
                    i32.const 1
                    i32.add
                    local.set 11
                    br 0 (;@8;)
                  end
                end
              end
              local.get 14
              local.get 17
              f64.lt
              if  ;; label = @6
                local.get 14
                local.set 17
                local.get 10
                i32.const 7
                i32.add
                local.set 15
              end
              local.get 32
              f64x2.extract_lane 1
              local.set 14
              local.get 14
              local.get 14
              f64.ne
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.gt
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.gt
              local.get 14
              f64.const 0x1p-1022 (;=2.22507e-308;)
              f64.lt
              i32.and
              i32.or
              i32.or
              if  ;; label = @6
                i32.const 1
                local.set 19
              end
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.eq
              if  ;; label = @6
                i32.const 0
                local.set 11
                block  ;; label = @7
                  loop  ;; label = @8
                    local.get 11
                    local.get 3
                    i32.ge_u
                    br_if 1 (;@7;)
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
                    local.get 1
                    local.get 11
                    local.get 5
                    i32.mul
                    local.get 10
                    i32.const 7
                    i32.add
                    i32.add
                    i32.const 8
                    i32.mul
                    i32.add
                    f64.load
                    f64.ne
                    if  ;; label = @9
                      i32.const 1
                      local.set 19
                      br 2 (;@7;)
                    end
                    local.get 11
                    i32.const 1
                    i32.add
                    local.set 11
                    br 0 (;@8;)
                  end
                end
              end
              local.get 14
              local.get 20
              f64.lt
              if  ;; label = @6
                local.get 14
                local.set 20
                local.get 10
                i32.const 7
                i32.add
                local.set 18
              end
            end
            local.get 10
            i32.const 8
            i32.add
            local.set 10
            br 0 (;@4;)
          end
        end
        local.get 6
        local.get 9
        i32.const 0
        i32.add
        i32.const 4
        i32.mul
        i32.add
        local.get 15
        i32.store
        local.get 7
        local.get 9
        i32.const 0
        i32.add
        i32.const 8
        i32.mul
        i32.add
        local.get 17
        f64.store
        local.get 8
        local.get 9
        i32.const 0
        i32.add
        i32.const 4
        i32.mul
        i32.add
        local.get 16
        i32.store
        local.get 6
        local.get 9
        i32.const 1
        i32.add
        i32.const 4
        i32.mul
        i32.add
        local.get 18
        i32.store
        local.get 7
        local.get 9
        i32.const 1
        i32.add
        i32.const 8
        i32.mul
        i32.add
        local.get 20
        f64.store
        local.get 8
        local.get 9
        i32.const 1
        i32.add
        i32.const 4
        i32.mul
        i32.add
        local.get 19
        i32.store
        local.get 9
        i32.const 2
        i32.add
        local.set 9
        br 0 (;@2;)
      end
    end
    local.get 9)
  (func (;1;) (type 0) (param i32 i32 i32 i32 i32 i32 i32 i32 i32) (result i32)
    (local i32 i32 i32 v128 v128 f64 i32 i32 f64 v128 v128 v128 v128 v128 v128 v128 v128)
    block  ;; label = @1
      loop  ;; label = @2
        local.get 9
        i32.const 1
        i32.add
        local.get 2
        i32.gt_u
        br_if 1 (;@1;)
        f64.const inf (;=inf;)
        local.set 17
        i32.const 0
        local.set 15
        i32.const 0
        local.set 16
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
              local.get 14
              f64.ne
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.gt
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.gt
              local.get 14
              f64.const 0x1p-1022 (;=2.22507e-308;)
              f64.lt
              i32.and
              i32.or
              i32.or
              if  ;; label = @6
                i32.const 1
                local.set 16
              end
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.eq
              if  ;; label = @6
                i32.const 0
                local.set 11
                block  ;; label = @7
                  loop  ;; label = @8
                    local.get 11
                    local.get 3
                    i32.ge_u
                    br_if 1 (;@7;)
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
                    f64.load
                    f64.ne
                    if  ;; label = @9
                      i32.const 1
                      local.set 16
                      br 2 (;@7;)
                    end
                    local.get 11
                    i32.const 1
                    i32.add
                    local.set 11
                    br 0 (;@8;)
                  end
                end
              end
              local.get 14
              local.get 17
              f64.lt
              if  ;; label = @6
                local.get 14
                local.set 17
                local.get 10
                i32.const 0
                i32.add
                local.set 15
              end
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
              local.get 14
              f64.ne
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.gt
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.gt
              local.get 14
              f64.const 0x1p-1022 (;=2.22507e-308;)
              f64.lt
              i32.and
              i32.or
              i32.or
              if  ;; label = @6
                i32.const 1
                local.set 16
              end
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.eq
              if  ;; label = @6
                i32.const 0
                local.set 11
                block  ;; label = @7
                  loop  ;; label = @8
                    local.get 11
                    local.get 3
                    i32.ge_u
                    br_if 1 (;@7;)
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
                    local.get 1
                    local.get 11
                    local.get 5
                    i32.mul
                    local.get 10
                    i32.const 1
                    i32.add
                    i32.add
                    i32.const 8
                    i32.mul
                    i32.add
                    f64.load
                    f64.ne
                    if  ;; label = @9
                      i32.const 1
                      local.set 16
                      br 2 (;@7;)
                    end
                    local.get 11
                    i32.const 1
                    i32.add
                    local.set 11
                    br 0 (;@8;)
                  end
                end
              end
              local.get 14
              local.get 17
              f64.lt
              if  ;; label = @6
                local.get 14
                local.set 17
                local.get 10
                i32.const 1
                i32.add
                local.set 15
              end
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
              local.get 14
              f64.ne
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.gt
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.gt
              local.get 14
              f64.const 0x1p-1022 (;=2.22507e-308;)
              f64.lt
              i32.and
              i32.or
              i32.or
              if  ;; label = @6
                i32.const 1
                local.set 16
              end
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.eq
              if  ;; label = @6
                i32.const 0
                local.set 11
                block  ;; label = @7
                  loop  ;; label = @8
                    local.get 11
                    local.get 3
                    i32.ge_u
                    br_if 1 (;@7;)
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
                    f64.load
                    f64.ne
                    if  ;; label = @9
                      i32.const 1
                      local.set 16
                      br 2 (;@7;)
                    end
                    local.get 11
                    i32.const 1
                    i32.add
                    local.set 11
                    br 0 (;@8;)
                  end
                end
              end
              local.get 14
              local.get 17
              f64.lt
              if  ;; label = @6
                local.get 14
                local.set 17
                local.get 10
                i32.const 2
                i32.add
                local.set 15
              end
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
              local.get 14
              f64.ne
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.gt
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.gt
              local.get 14
              f64.const 0x1p-1022 (;=2.22507e-308;)
              f64.lt
              i32.and
              i32.or
              i32.or
              if  ;; label = @6
                i32.const 1
                local.set 16
              end
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.eq
              if  ;; label = @6
                i32.const 0
                local.set 11
                block  ;; label = @7
                  loop  ;; label = @8
                    local.get 11
                    local.get 3
                    i32.ge_u
                    br_if 1 (;@7;)
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
                    local.get 1
                    local.get 11
                    local.get 5
                    i32.mul
                    local.get 10
                    i32.const 3
                    i32.add
                    i32.add
                    i32.const 8
                    i32.mul
                    i32.add
                    f64.load
                    f64.ne
                    if  ;; label = @9
                      i32.const 1
                      local.set 16
                      br 2 (;@7;)
                    end
                    local.get 11
                    i32.const 1
                    i32.add
                    local.set 11
                    br 0 (;@8;)
                  end
                end
              end
              local.get 14
              local.get 17
              f64.lt
              if  ;; label = @6
                local.get 14
                local.set 17
                local.get 10
                i32.const 3
                i32.add
                local.set 15
              end
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
              local.get 14
              f64.ne
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.gt
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.gt
              local.get 14
              f64.const 0x1p-1022 (;=2.22507e-308;)
              f64.lt
              i32.and
              i32.or
              i32.or
              if  ;; label = @6
                i32.const 1
                local.set 16
              end
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.eq
              if  ;; label = @6
                i32.const 0
                local.set 11
                block  ;; label = @7
                  loop  ;; label = @8
                    local.get 11
                    local.get 3
                    i32.ge_u
                    br_if 1 (;@7;)
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
                    f64.load
                    f64.ne
                    if  ;; label = @9
                      i32.const 1
                      local.set 16
                      br 2 (;@7;)
                    end
                    local.get 11
                    i32.const 1
                    i32.add
                    local.set 11
                    br 0 (;@8;)
                  end
                end
              end
              local.get 14
              local.get 17
              f64.lt
              if  ;; label = @6
                local.get 14
                local.set 17
                local.get 10
                i32.const 4
                i32.add
                local.set 15
              end
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
              local.get 14
              f64.ne
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.gt
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.gt
              local.get 14
              f64.const 0x1p-1022 (;=2.22507e-308;)
              f64.lt
              i32.and
              i32.or
              i32.or
              if  ;; label = @6
                i32.const 1
                local.set 16
              end
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.eq
              if  ;; label = @6
                i32.const 0
                local.set 11
                block  ;; label = @7
                  loop  ;; label = @8
                    local.get 11
                    local.get 3
                    i32.ge_u
                    br_if 1 (;@7;)
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
                    local.get 1
                    local.get 11
                    local.get 5
                    i32.mul
                    local.get 10
                    i32.const 5
                    i32.add
                    i32.add
                    i32.const 8
                    i32.mul
                    i32.add
                    f64.load
                    f64.ne
                    if  ;; label = @9
                      i32.const 1
                      local.set 16
                      br 2 (;@7;)
                    end
                    local.get 11
                    i32.const 1
                    i32.add
                    local.set 11
                    br 0 (;@8;)
                  end
                end
              end
              local.get 14
              local.get 17
              f64.lt
              if  ;; label = @6
                local.get 14
                local.set 17
                local.get 10
                i32.const 5
                i32.add
                local.set 15
              end
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
              local.get 14
              f64.ne
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.gt
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.gt
              local.get 14
              f64.const 0x1p-1022 (;=2.22507e-308;)
              f64.lt
              i32.and
              i32.or
              i32.or
              if  ;; label = @6
                i32.const 1
                local.set 16
              end
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.eq
              if  ;; label = @6
                i32.const 0
                local.set 11
                block  ;; label = @7
                  loop  ;; label = @8
                    local.get 11
                    local.get 3
                    i32.ge_u
                    br_if 1 (;@7;)
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
                    f64.load
                    f64.ne
                    if  ;; label = @9
                      i32.const 1
                      local.set 16
                      br 2 (;@7;)
                    end
                    local.get 11
                    i32.const 1
                    i32.add
                    local.set 11
                    br 0 (;@8;)
                  end
                end
              end
              local.get 14
              local.get 17
              f64.lt
              if  ;; label = @6
                local.get 14
                local.set 17
                local.get 10
                i32.const 6
                i32.add
                local.set 15
              end
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
              local.get 14
              f64.ne
              local.get 14
              f64.const 0x1.fffffffffffffp+1023 (;=1.79769e+308;)
              f64.gt
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.gt
              local.get 14
              f64.const 0x1p-1022 (;=2.22507e-308;)
              f64.lt
              i32.and
              i32.or
              i32.or
              if  ;; label = @6
                i32.const 1
                local.set 16
              end
              local.get 14
              f64.const 0x0p+0 (;=0;)
              f64.eq
              if  ;; label = @6
                i32.const 0
                local.set 11
                block  ;; label = @7
                  loop  ;; label = @8
                    local.get 11
                    local.get 3
                    i32.ge_u
                    br_if 1 (;@7;)
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
                    local.get 1
                    local.get 11
                    local.get 5
                    i32.mul
                    local.get 10
                    i32.const 7
                    i32.add
                    i32.add
                    i32.const 8
                    i32.mul
                    i32.add
                    f64.load
                    f64.ne
                    if  ;; label = @9
                      i32.const 1
                      local.set 16
                      br 2 (;@7;)
                    end
                    local.get 11
                    i32.const 1
                    i32.add
                    local.set 11
                    br 0 (;@8;)
                  end
                end
              end
              local.get 14
              local.get 17
              f64.lt
              if  ;; label = @6
                local.get 14
                local.set 17
                local.get 10
                i32.const 7
                i32.add
                local.set 15
              end
            end
            local.get 10
            i32.const 8
            i32.add
            local.set 10
            br 0 (;@4;)
          end
        end
        local.get 6
        local.get 9
        i32.const 0
        i32.add
        i32.const 4
        i32.mul
        i32.add
        local.get 15
        i32.store
        local.get 7
        local.get 9
        i32.const 0
        i32.add
        i32.const 8
        i32.mul
        i32.add
        local.get 17
        f64.store
        local.get 8
        local.get 9
        i32.const 0
        i32.add
        i32.const 4
        i32.mul
        i32.add
        local.get 16
        i32.store
        local.get 9
        i32.const 1
        i32.add
        local.set 9
        br 0 (;@2;)
      end
    end
    local.get 9)
  (func (;2;) (type 1) (param i32 i32 i32 i32 i32 i32 i32 i32 i32)
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
    call 0
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
    i32.const 4
    i32.mul
    i32.add
    local.get 7
    local.get 9
    i32.const 8
    i32.mul
    i32.add
    local.get 8
    local.get 9
    i32.const 4
    i32.mul
    i32.add
    call 1
    drop)
  (memory (;0;) 1)
  (export "memory" (memory 0))
  (export "nearest" (func 2)))

