;; RECONSTRUCTED FROM BINARY, not recovered original source.
;; Input SHA-256: 57bdca8ae332945e3f56cf4bfdf6178dee1020f237991aa5810a8489a680d6a9
;; Tool: wabt 1.0.39; readWasm(readDebugNames:true,simd:true), applyNames, toText.
;; Original comments, source formatting, authorship and license are NOT recovered.
(module
  (type (;0;) (func (param i32 i32 i32 i32 i32 f64 f64)))
  (func (;0;) (type 0) (param i32 i32 i32 i32 i32 f64 f64)
    (local i32 i32 i32 f64 f64 f64 f64 v128 v128 v128 v128)
    local.get 0
    local.get 3
    local.get 2
    i32.mul
    i32.const 8
    i32.mul
    i32.add
    local.set 8
    local.get 0
    local.get 4
    local.get 2
    i32.mul
    i32.const 8
    i32.mul
    i32.add
    local.set 9
    block  ;; label = @1
      loop  ;; label = @2
        local.get 7
        local.get 2
        i32.ge_u
        br_if 1 (;@1;)
        local.get 7
        local.get 3
        i32.ne
        local.get 7
        local.get 4
        i32.ne
        i32.and
        if  ;; label = @3
          local.get 8
          local.get 7
          i32.const 8
          i32.mul
          i32.add
          f64.load
          local.set 10
          local.get 9
          local.get 7
          i32.const 8
          i32.mul
          i32.add
          f64.load
          local.set 11
          local.get 5
          local.get 10
          f64.mul
          local.get 6
          local.get 11
          f64.mul
          f64.sub
          local.set 12
          local.get 6
          local.get 10
          f64.mul
          local.get 5
          local.get 11
          f64.mul
          f64.add
          local.set 13
          local.get 8
          local.get 7
          i32.const 8
          i32.mul
          i32.add
          local.get 12
          f64.store
          local.get 0
          local.get 7
          local.get 2
          i32.mul
          local.get 3
          i32.add
          i32.const 8
          i32.mul
          i32.add
          local.get 12
          f64.store
          local.get 9
          local.get 7
          i32.const 8
          i32.mul
          i32.add
          local.get 13
          f64.store
          local.get 0
          local.get 7
          local.get 2
          i32.mul
          local.get 4
          i32.add
          i32.const 8
          i32.mul
          i32.add
          local.get 13
          f64.store
        end
        local.get 7
        i32.const 1
        i32.add
        local.set 7
        br 0 (;@2;)
      end
    end
    i32.const 0
    local.set 7
    local.get 1
    local.get 3
    local.get 2
    i32.mul
    i32.const 8
    i32.mul
    i32.add
    local.set 8
    local.get 1
    local.get 4
    local.get 2
    i32.mul
    i32.const 8
    i32.mul
    i32.add
    local.set 9
    local.get 5
    f64x2.splat
    local.set 14
    local.get 6
    f64x2.splat
    local.set 15
    block  ;; label = @1
      loop  ;; label = @2
        local.get 7
        i32.const 1
        i32.add
        local.get 2
        i32.ge_u
        br_if 1 (;@1;)
        local.get 8
        local.get 7
        i32.const 8
        i32.mul
        i32.add
        v128.load
        local.set 16
        local.get 9
        local.get 7
        i32.const 8
        i32.mul
        i32.add
        v128.load
        local.set 17
        local.get 8
        local.get 7
        i32.const 8
        i32.mul
        i32.add
        local.get 14
        local.get 16
        f64x2.mul
        local.get 15
        local.get 17
        f64x2.mul
        f64x2.sub
        v128.store
        local.get 9
        local.get 7
        i32.const 8
        i32.mul
        i32.add
        local.get 15
        local.get 16
        f64x2.mul
        local.get 14
        local.get 17
        f64x2.mul
        f64x2.add
        v128.store
        local.get 7
        i32.const 2
        i32.add
        local.set 7
        br 0 (;@2;)
      end
    end
    local.get 7
    local.get 2
    i32.lt_u
    if  ;; label = @1
      local.get 8
      local.get 7
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set 10
      local.get 9
      local.get 7
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set 11
      local.get 8
      local.get 7
      i32.const 8
      i32.mul
      i32.add
      local.get 5
      local.get 10
      f64.mul
      local.get 6
      local.get 11
      f64.mul
      f64.sub
      f64.store
      local.get 9
      local.get 7
      i32.const 8
      i32.mul
      i32.add
      local.get 6
      local.get 10
      f64.mul
      local.get 5
      local.get 11
      f64.mul
      f64.add
      f64.store
    end)
  (memory (;0;) 1)
  (export "memory" (memory 0))
  (export "rotate" (func 0)))

