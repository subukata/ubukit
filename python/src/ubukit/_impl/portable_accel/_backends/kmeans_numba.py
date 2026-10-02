"""Strict adaptive-center/four-row register blocking via LLVM vector intrinsics.

The intrinsic is target-independent LLVM vector IR, not embedded x86 assembly:
LLVM may lower it to AVX-512, split narrower SIMD, or scalar instructions.
"""
import numpy as np
from llvmlite import ir
from numba import njit, prange, types, config
from numba.core import cgutils
from numba.extending import intrinsic
from numba.core.errors import TypingError


@intrinsic
def _nearest8(typing_context, x_type, row_type, centers_type, start_type):
    """Return nearest squared distance, second nearest, and local center index.

    Caller guarantees eight valid (possibly padded) consecutive center lanes.
    """
    if not (isinstance(x_type, types.Array) and x_type.ndim == 2
            and x_type.dtype == types.float64 and x_type.layout == 'C'
            and isinstance(centers_type, types.Array) and centers_type.ndim == 2
            and centers_type.dtype == types.float64 and centers_type.layout == 'C'
            and isinstance(row_type, types.Integer) and isinstance(start_type, types.Integer)):
        raise TypingError('distance selector requires contiguous float64 matrices and integer indices')
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
            xvalue = builder.load(xptr, align=1)
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


@intrinsic
def _nearest8rows4(typing_context, x_type, row_type, centers_type, start_type):
    """Four independent rows, exact increasing-feature sums in each vector.

    Caller guarantees four input rows and eight (possibly padded) center lanes.
    """
    if not (isinstance(x_type, types.Array) and x_type.ndim == 2
            and x_type.dtype == types.float64 and x_type.layout == 'C'
            and isinstance(centers_type, types.Array) and centers_type.ndim == 2
            and centers_type.dtype == types.float64 and centers_type.layout == 'C'
            and isinstance(row_type, types.Integer) and isinstance(start_type, types.Integer)):
        raise TypingError('distance selector requires contiguous float64 matrices and integer indices')
    output_type = types.Tuple((types.float64,) * 4 + (types.int64,) * 4)
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
        accumulators = [cgutils.alloca_once(builder, vector_type) for _ in range(4)]
        for accumulator in accumulators:
            builder.store(zero, accumulator)
        with cgutils.for_range(builder, dimension) as loop:
            center_ptr = cgutils.get_item_pointer(
                context, builder, sig.args[2], ct, [loop.index, args[3]])
            center_vector_ptr = builder.bitcast(center_ptr, vector_type.as_pointer())
            center_values = builder.load(center_vector_ptr, align=8)
            for row_delta, accumulator in enumerate(accumulators):
                row = builder.add(args[1], ir.Constant(args[1].type, row_delta))
                xptr = cgutils.get_item_pointer(
                    context, builder, sig.args[0], x, [row, loop.index])
                xvalue = builder.load(xptr, align=1)
                broadcast = builder.shuffle_vector(
                    builder.insert_element(undef, xvalue, i32(0)), undef, mask)
                delta = builder.fsub(broadcast, center_values)
                squared = builder.fmul(delta, delta)
                value = builder.fadd(builder.load(accumulator), squared)
                builder.store(value, accumulator)
        bests, indices = [], []
        for accumulator in accumulators:
            distances = builder.load(accumulator)
            best = ir.Constant(scalar, float('inf'))
            index = i64(0)
            for lane in range(8):
                distance = builder.extract_element(distances, i32(lane))
                less = builder.fcmp_ordered('<', distance, best)
                best = builder.select(less, distance, best)
                index = builder.select(less, i64(lane), index)
            bests.append(best)
            indices.append(index)
        return context.make_tuple(builder, sig.return_type, tuple(bests + indices))

    return signature, codegen


@intrinsic
def _nearest4rows4(typing_context, x_type, row_type, centers_type, start_type):
    """Four independent rows, exact increasing-feature sums in each vector.

    Caller guarantees four input rows and four (possibly padded) center lanes.
    """
    if not (isinstance(x_type, types.Array) and x_type.ndim == 2
            and x_type.dtype == types.float64 and x_type.layout == 'C'
            and isinstance(centers_type, types.Array) and centers_type.ndim == 2
            and centers_type.dtype == types.float64 and centers_type.layout == 'C'
            and isinstance(row_type, types.Integer) and isinstance(start_type, types.Integer)):
        raise TypingError('distance selector requires contiguous float64 matrices and integer indices')
    output_type = types.Tuple((types.float64,) * 4 + (types.int64,) * 4)
    signature = output_type(x_type, row_type, centers_type, start_type)

    def codegen(context, builder, sig, args):
        x = context.make_array(sig.args[0])(context, builder, value=args[0])
        ct = context.make_array(sig.args[2])(context, builder, value=args[2])
        dimension = cgutils.unpack_tuple(builder, x.shape, count=2)[1]
        scalar = ir.DoubleType()
        i32 = ir.IntType(32)
        i64 = ir.IntType(64)
        vector_type = ir.VectorType(scalar, 4)
        zero = ir.Constant(vector_type, [0.0] * 4)
        undef = ir.Constant(vector_type, ir.Undefined)
        mask = ir.Constant(ir.VectorType(i32, 4), [0] * 4)
        accumulators = [cgutils.alloca_once(builder, vector_type) for _ in range(4)]
        for accumulator in accumulators:
            builder.store(zero, accumulator)
        with cgutils.for_range(builder, dimension) as loop:
            center_ptr = cgutils.get_item_pointer(
                context, builder, sig.args[2], ct, [loop.index, args[3]])
            center_vector_ptr = builder.bitcast(center_ptr, vector_type.as_pointer())
            center_values = builder.load(center_vector_ptr, align=8)
            for row_delta, accumulator in enumerate(accumulators):
                row = builder.add(args[1], ir.Constant(args[1].type, row_delta))
                xptr = cgutils.get_item_pointer(
                    context, builder, sig.args[0], x, [row, loop.index])
                xvalue = builder.load(xptr, align=1)
                broadcast = builder.shuffle_vector(
                    builder.insert_element(undef, xvalue, i32(0)), undef, mask)
                delta = builder.fsub(broadcast, center_values)
                squared = builder.fmul(delta, delta)
                value = builder.fadd(builder.load(accumulator), squared)
                builder.store(value, accumulator)
        bests, indices = [], []
        for accumulator in accumulators:
            distances = builder.load(accumulator)
            best = ir.Constant(scalar, float('inf'))
            index = i64(0)
            for lane in range(4):
                distance = builder.extract_element(distances, i32(lane))
                less = builder.fcmp_ordered('<', distance, best)
                best = builder.select(less, distance, best)
                index = builder.select(less, i64(lane), index)
            bests.append(best)
            indices.append(index)
        return context.make_tuple(builder, sig.return_type, tuple(bests + indices))

    return signature, codegen


@intrinsic
def _nearest16rows4(typing_context, x_type, row_type, centers_type, start_type):
    """Four independent rows, exact increasing-feature sums in each vector.

    Caller guarantees four input rows and sixteen (possibly padded) center lanes.
    """
    if not (isinstance(x_type, types.Array) and x_type.ndim == 2
            and x_type.dtype == types.float64 and x_type.layout == 'C'
            and isinstance(centers_type, types.Array) and centers_type.ndim == 2
            and centers_type.dtype == types.float64 and centers_type.layout == 'C'
            and isinstance(row_type, types.Integer) and isinstance(start_type, types.Integer)):
        raise TypingError('distance selector requires contiguous float64 matrices and integer indices')
    output_type = types.Tuple((types.float64,) * 4 + (types.int64,) * 4)
    signature = output_type(x_type, row_type, centers_type, start_type)

    def codegen(context, builder, sig, args):
        x = context.make_array(sig.args[0])(context, builder, value=args[0])
        ct = context.make_array(sig.args[2])(context, builder, value=args[2])
        dimension = cgutils.unpack_tuple(builder, x.shape, count=2)[1]
        scalar = ir.DoubleType()
        i32 = ir.IntType(32)
        i64 = ir.IntType(64)
        vector_type = ir.VectorType(scalar, 16)
        zero = ir.Constant(vector_type, [0.0] * 16)
        undef = ir.Constant(vector_type, ir.Undefined)
        mask = ir.Constant(ir.VectorType(i32, 16), [0] * 16)
        accumulators = [cgutils.alloca_once(builder, vector_type) for _ in range(4)]
        for accumulator in accumulators:
            builder.store(zero, accumulator)
        with cgutils.for_range(builder, dimension) as loop:
            center_ptr = cgutils.get_item_pointer(
                context, builder, sig.args[2], ct, [loop.index, args[3]])
            center_vector_ptr = builder.bitcast(center_ptr, vector_type.as_pointer())
            center_values = builder.load(center_vector_ptr, align=8)
            for row_delta, accumulator in enumerate(accumulators):
                row = builder.add(args[1], ir.Constant(args[1].type, row_delta))
                xptr = cgutils.get_item_pointer(
                    context, builder, sig.args[0], x, [row, loop.index])
                xvalue = builder.load(xptr, align=1)
                broadcast = builder.shuffle_vector(
                    builder.insert_element(undef, xvalue, i32(0)), undef, mask)
                delta = builder.fsub(broadcast, center_values)
                squared = builder.fmul(delta, delta)
                value = builder.fadd(builder.load(accumulator), squared)
                builder.store(value, accumulator)
        bests, indices = [], []
        for accumulator in accumulators:
            distances = builder.load(accumulator)
            best = ir.Constant(scalar, float('inf'))
            index = i64(0)
            for lane in range(16):
                distance = builder.extract_element(distances, i32(lane))
                less = builder.fcmp_ordered('<', distance, best)
                best = builder.select(less, distance, best)
                index = builder.select(less, i64(lane), index)
            bests.append(best)
            indices.append(index)
        return context.make_tuple(builder, sig.return_type, tuple(bests + indices))

    return signature, codegen


@njit(cache=True, parallel=True)
def _register8(X, init, max_iter, blocks, vector_width=8):
    n, d = X.shape
    k = init.shape[0]
    centers = init.copy()
    if vector_width not in (4, 8, 16):
        raise ValueError("vector width must be 4, 8, or 16")
    if k == 1:
        # The sole center wins every assignment. Preserve the original row
        # partitions, increasing-row sums, merge order, and convergence count.
        labels = np.zeros(n, dtype=np.int64)
        partial = np.empty((blocks, d), dtype=np.float64)
        for b in prange(blocks):
            partial[b].fill(0.0)
            for i in range(n * b // blocks, n * (b + 1) // blocks):
                for j in range(d):
                    partial[b, j] += X[i, j]
        for j in range(d):
            total = 0.0
            for b in range(blocks):
                total += partial[b, j]
            centers[0, j] = total / n
        return centers, labels, min(max_iter, 2)
    padded_k = max(((k + 7) // 8) * 8, ((k + vector_width - 1) // vector_width) * vector_width)
    centers_t = np.full((d, padded_k), np.inf, dtype=np.float64)
    labels = np.full(n, -1, dtype=np.int64)
    sums = np.empty((blocks, k, d), dtype=np.float64)
    counts = np.empty((blocks, k), dtype=np.int64)
    changes = np.empty(blocks, dtype=np.int64)
    iteration = 0
    for iteration in range(1, max_iter + 1):
        # Contiguous destination writes avoid repeated strided cache-line
        # ownership traffic when D or K is large; this is a pure copy.
        for j in range(d):
            for c in range(k):
                centers_t[j, c] = centers[c, j]
        for b in prange(blocks):
            sums[b].fill(0.0)
            counts[b].fill(0)
            changed = 0
            first, stop = n * b // blocks, n * (b + 1) // blocks
            for i in range(first, stop - 3, 4):
                d0, d1, d2, d3 = np.inf, np.inf, np.inf, np.inf
                b0, b1, b2, b3 = 0, 0, 0, 0
                for c in range(0, k, vector_width):
                    if vector_width == 4:
                        a0,a1,a2,a3,l0,l1,l2,l3 = _nearest4rows4(X, i, centers_t, c)
                    elif vector_width == 16:
                        a0,a1,a2,a3,l0,l1,l2,l3 = _nearest16rows4(X, i, centers_t, c)
                    else:
                        a0,a1,a2,a3,l0,l1,l2,l3 = _nearest8rows4(X, i, centers_t, c)
                    if a0 < d0: d0, b0 = a0, c + l0
                    if a1 < d1: d1, b1 = a1, c + l1
                    if a2 < d2: d2, b2 = a2, c + l2
                    if a3 < d3: d3, b3 = a3, c + l3
                chosen = (b0, b1, b2, b3)
                for offset in range(4):
                    row = i + offset
                    best = chosen[offset]
                    if labels[row] != best:
                        changed = 1
                        labels[row] = best
                    counts[b, best] += 1
                    for j in range(d):
                        sums[b, best, j] += X[row, j]
            for i in range(first + ((stop-first)//4)*4, stop):
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




def _host_vector_cap():
    # Wide target-neutral IR is correct on narrower machines, but a 4x16 tile
    # can spill registers there. Use it only on a native AVX-512 target; keep
    # a conservative four-lane route for explicit/generic CPU configurations.
    try:
        from llvmlite.binding import get_host_cpu_features
        features = get_host_cpu_features()
        if features.get('avx512f', False):
            return 16
        if features.get('avx2', False) or features.get('neon', False):
            return 8
    except (RuntimeError, AttributeError):
        pass
    return 4


_HOST_VECTOR_CAP = _host_vector_cap()


def _runtime_vector_cap():
    return (4 if config.CPU_NAME is not None or config.CPU_FEATURES is not None
            else _HOST_VECTOR_CAP)


def _select_vector_width(k, d):
    """Avoid padding and register spills; respect explicit Numba CPU targets."""
    cap = _runtime_vector_cap()
    if k <= 4 or cap == 4:
        return 4
    padded8 = ((k + 7) // 8) * 8
    padded16 = ((k + 15) // 16) * 16
    return 16 if cap >= 16 and d >= 8 and 10 * padded16 <= 11 * padded8 else 8
