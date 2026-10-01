"""Strict eight-center register blocking via an explicit LLVM vector intrinsic.

The intrinsic is target-independent LLVM vector IR, not embedded x86 assembly:
LLVM may lower it to AVX-512, split narrower SIMD, or scalar instructions.
"""
import numpy as np
from llvmlite import ir
from numba import njit, prange, types
from numba.core import cgutils
from numba.extending import intrinsic


@intrinsic
def _nearest8(typing_context, x_type, row_type, centers_type, start_type):
    """Return nearest squared distance, second nearest, and local center index.

    Caller guarantees C-contiguous float64 arrays and eight valid consecutive
    centers in the second dimension of the transposed center array.
    """
    output_type = types.Tuple((types.float64, types.float64, types.int64))
    signature = output_type(x_type, row_type, centers_type, start_type)

    def codegen(context, builder, sig, args):
        x = context.make_array(sig.args[0])(context, builder, value=args[0])
        ct = context.make_array(sig.args[2])(context, builder, value=args[2])
        dimension = cgutils.unpack_tuple(builder, x.shape, count=2)[1]
        scalar = ir.DoubleType()
        i32 = ir.IntType(32)
        i64 = ir.IntType(64)
        vector_type = ir.VectorType(scalar, 8)
        zero = ir.Constant(vector_type, [0.0] * 8)
        undef = ir.Constant(vector_type, ir.Undefined)
        mask = ir.Constant(ir.VectorType(i32, 8), [0] * 8)
        accumulator = cgutils.alloca_once(builder, vector_type)
        builder.store(zero, accumulator)
        with cgutils.for_range(builder, dimension) as loop:
            xptr = cgutils.get_item_pointer(
                context, builder, sig.args[0], x, [args[1], loop.index])
            xvalue = builder.load(xptr)
            broadcast = builder.shuffle_vector(
                builder.insert_element(undef, xvalue, i32(0)), undef, mask)
            center_ptr = cgutils.get_item_pointer(
                context, builder, sig.args[2], ct, [loop.index, args[3]])
            center_vector_ptr = builder.bitcast(center_ptr, vector_type.as_pointer())
            center_values = builder.load(center_vector_ptr, align=8)
            delta = builder.fsub(broadcast, center_values)
            squared = builder.fmul(delta, delta)
            value = builder.fadd(builder.load(accumulator), squared)
            builder.store(value, accumulator)
        distances = builder.load(accumulator)
        best = ir.Constant(scalar, float('inf'))
        second = ir.Constant(scalar, float('inf'))
        index = i64(0)
        # Ascending lane order and strictly-less comparisons guarantee first ties.
        for lane in range(8):
            distance = builder.extract_element(distances, i32(lane))
            less = builder.fcmp_ordered('<', distance, best)
            second_less = builder.fcmp_ordered('<', distance, second)
            second = builder.select(less, best, builder.select(second_less, distance, second))
            best = builder.select(less, distance, best)
            index = builder.select(less, i64(lane), index)
        return context.make_tuple(builder, sig.return_type, (best, second, index))

    return signature, codegen


@njit(cache=True, parallel=True)
def _register8(X, init, max_iter, blocks):
    n, d = X.shape
    k = init.shape[0]
    centers = init.copy()
    padded_k = ((k + 7) // 8) * 8
    centers_t = np.full((d, padded_k), np.inf, dtype=np.float64)
    labels = np.full(n, -1, dtype=np.int64)
    sums = np.empty((blocks, k, d), dtype=np.float64)
    counts = np.empty((blocks, k), dtype=np.int64)
    changes = np.empty(blocks, dtype=np.int64)
    iteration = 0
    for iteration in range(1, max_iter + 1):
        for c in range(k):
            for j in range(d):
                centers_t[j, c] = centers[c, j]
        for b in prange(blocks):
            sums[b].fill(0.0)
            counts[b].fill(0)
            changed = 0
            for i in range(n * b // blocks, n * (b + 1) // blocks):
                best = 0
                best_dist = np.inf
                for c in range(0, k, 8):
                    distance, unused, lane = _nearest8(X, i, centers_t, c)
                    if distance < best_dist:
                        best_dist = distance
                        best = c + lane
                if labels[i] != best:
                    changed = 1
                    labels[i] = best
                counts[b, best] += 1
                for j in range(d):
                    sums[b, best, j] += X[i, j]
            changes[b] = changed
        for c in range(k):
            count = 0
            for b in range(blocks):
                count += counts[b, c]
            if count != 0:
                for j in range(d):
                    total = 0.0
                    for b in range(blocks):
                        total += sums[b, c, j]
                    centers[c, j] = total / count
        changed = False
        for b in range(blocks):
            if changes[b] != 0:
                changed = True
        if not changed:
            break
    return centers, labels, iteration

