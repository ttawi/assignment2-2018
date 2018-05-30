from __future__ import absolute_import, print_function

import tvm
import numpy as np
import topi

# Global declarations of environment.

# llvm
tgt_host = "llvm"
# llvm, cuda, opencl, metal
# Change it to respective GPU if gpu is enabled Ex: cuda, opencl
tgt = "llvm"


def make_elemwise_add(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.placeholder(shape, dtype=dtype, name="B")
    C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_elemwise_mul(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.placeholder(shape, dtype=dtype, name="B")
    C = tvm.compute(A.shape, lambda *i: A(*i) * B(*i))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_elemwise_add_by_const(shape, const_k, tgt, tgt_host, func_name,
                               dtype="float32"):
    """TODO: Your code here"""
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = tvm.compute(A.shape, lambda *i: A(*i) + tvm.const(const_k, dtype=dtype))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_elemwise_mul_by_const(shape, const_k, tgt, tgt_host, func_name,
                               dtype="float32"):
    """TODO: Your code here"""
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = tvm.compute(A.shape, lambda *i: A(*i) * tvm.const(const_k, dtype))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_relu(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.max, tvm.const(0, A.dtype)"""
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = tvm.compute(A.shape, lambda *i: tvm.max(A(*i), tvm.const(0, A.dtype)))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_relu_gradient(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.select"""
    #   gradient of relu: https://www.quora.com/How-do-we-compute-the-gradient-of-a-ReLU-for-backpropagation
    #   this function calculates the gradient for relu node during back prop. It equals to 1/0 * the previous
    #   gradient (B)
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.placeholder(shape, dtype=dtype, name="B")
    C = tvm.compute(A.shape, lambda *i: tvm.convert(tvm.select(A(*i) <= 0, tvm.const(0, dtype), B(*i))))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_matrix_mul(shapeA, transposeA, shapeB, transposeB, tgt, tgt_host,
                    func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum"""
    """Hint: treat 4 cases of transposeA, transposeB separately"""
    """Hint: for tvm schedule, use split, reorder, vectorize, parallel"""
    """Hint: debug tvm schedule using tvm.lower"""
    # tile is tile_size * split_size
    tile_size = 32
    split_size = 4

    A = tvm.placeholder(shapeA, dtype=dtype, name="A")
    B = tvm.placeholder(shapeB, dtype=dtype, name="B")
    packedA = None
    packedB = None

    if not transposeA:
        if not transposeB:
            # optimize the structure of B
            packedB = tvm.compute((shapeB[1] / tile_size, shapeB[0], tile_size),
                                  lambda x, y, z: B[y, x * tile_size + z], name="packedB")
            Aj = tvm.reduce_axis((0, A.shape[1]), "Aj")
            Output = tvm.compute(
                (A.shape[0], B.shape[1]),
                lambda i, j: tvm.sum(A[i, Aj] * packedB[j / tile_size, Aj, j % tile_size], axis=[Aj]),
                name="Output")
            s = tvm.create_schedule(Output.op)

            # Write cache for blocks
            CC = s.cache_write(Output, 'global')

            # Optimize schedule by blocking
            xo, yo, xi, yi = s[Output].tile(Output.op.axis[0], Output.op.axis[1], x_factor=tile_size,
                                            y_factor=tile_size)

            # Write cache computed at yo
            s[CC].compute_at(s[Output], yo)

            # New inner axes
            xc, yc = s[CC].op.axis

            k, = s[CC].op.reduce_axis
            ko, ki = s[CC].split(k, factor=split_size)

            # Re-order permutation
            # s[Output].reorder(xo, yo, ko, xi, ki, yi)
            # s[Output].vectorize(yi)
            s[CC].reorder(ko, xc, ki, yc)
            s[CC].unroll(ki)
            s[CC].vectorize(yc)
        else:
            Aj = tvm.reduce_axis((0, A.shape[1]), "Aj")
            Output = tvm.compute(
                (A.shape[0], B.shape[0]),
                lambda i, j: tvm.sum(A[i, Aj] * B[j, Aj], axis=[Aj]),
                name="Output")

            s = tvm.create_schedule(Output.op)

            # Write cache for blocks
            CC = s.cache_write(Output, 'global')

            # Optimize schedule by blocking
            xo, yo, xi, yi = s[Output].tile(Output.op.axis[0], Output.op.axis[1], x_factor=tile_size,
                                            y_factor=tile_size)

            # Write cache computed at yo
            s[CC].compute_at(s[Output], yo)

            # New inner axes
            xc, yc = s[CC].op.axis

            k, = s[CC].op.reduce_axis
            ko, ki = s[CC].split(k, factor=split_size)

            # Re-order permutation
            # s[Output].reorder(xo, yo, ko, xi, yi, ki)
            # s[Output].vectorize(ki)
            s[CC].reorder(ko, xc, yc, ki)
            s[CC].unroll(ki)

    else:
        if not transposeB:
            packedA = tvm.compute((shapeA[1] / tile_size, shapeA[0], tile_size),
                                  lambda x, y, z: A[y, x * tile_size + z], name="packedA")
            packedB = tvm.compute((shapeB[1] / tile_size, shapeB[0], tile_size),
                                  lambda x, y, z: B[y, x * tile_size + z], name="packedB")
            Aj = tvm.reduce_axis((0, A.shape[0]), "Aj")  # Ai in ori
            Output = tvm.compute(
                (A.shape[1], B.shape[1]),
                # lambda i, j: tvm.sum(A[Aj, i] * packedB[j / tile_size, Aj, j % tile_size], axis=[Aj]),
                lambda i, j: tvm.sum(packedA[i / tile_size, Aj, i % tile_size] * packedB[j / tile_size, Aj, j % tile_size], axis=[Aj]),
                name="Output")

            s = tvm.create_schedule(Output.op)

            # Write cache for blocks
            CC = s.cache_write(Output, 'global')

            # Optimize schedule by blocking
            xo, yo, xi, yi = s[Output].tile(Output.op.axis[0], Output.op.axis[1], x_factor=tile_size,
                                            y_factor=tile_size)

            # Write cache computed at yo
            s[CC].compute_at(s[Output], yo)

            # New inner axes
            xc, yc = s[CC].op.axis

            k, = s[CC].op.reduce_axis
            ko, ki = s[CC].split(k, factor=split_size)

            # Re-order permutation
            # s[Output].reorder(xo, yo, ko, ki, xi, yi)
            # s[Output].vectorize(yi)
            s[CC].reorder(ko, ki, xc, yc)
            s[CC].unroll(ki)
            s[CC].vectorize(yc)
        else:
            packedA = tvm.compute((shapeA[1] / tile_size, shapeA[0], tile_size),
                                  lambda x, y, z: A[y, x * tile_size + z], name="packedA")
            Aj = tvm.reduce_axis((0, A.shape[0]), "Aj")  # Ai in ori
            Output = tvm.compute(
                (A.shape[1], B.shape[0]),
                lambda i, j: tvm.sum(packedA[i / tile_size, Aj, i % tile_size] * B[j, Aj], axis=[Aj]),
                name="Output")

            s = tvm.create_schedule(Output.op)

            # Write cache for blocks
            CC = s.cache_write(Output, 'global')

            # Optimize schedule by blocking
            xo, yo, xi, yi = s[Output].tile(Output.op.axis[0], Output.op.axis[1], x_factor=tile_size,
                                            y_factor=tile_size)

            # Write cache computed at yo
            s[CC].compute_at(s[Output], yo)

            # New inner axes
            xc, yc = s[CC].op.axis

            k, = s[CC].op.reduce_axis
            ko, ki = s[CC].split(k, factor=split_size)

            # Re-order permutation
            # s[Output].reorder(xo, yo, ko, yi, ki, xi)
            # s[Output].vectorize(xi)
            s[CC].reorder(ko, yc, ki, xc)
            s[CC].unroll(ki)
            s[CC].vectorize(xc)

    # Parallel
    s[Output].parallel(xo)

    # Array Packing
    if packedA is not None:
        x, y, z = s[packedA].op.axis
        s[packedA].vectorize(z)
        s[packedA].parallel(x)
    if packedB is not None:
        x, y, z = s[packedB].op.axis
        s[packedB].vectorize(z)
        s[packedB].parallel(x)

    # print (tvm.lower(s, [A, B, Output], simple_mode=True))
    f = tvm.build(s, [A, B, Output], tgt, target_host=tgt_host, name=func_name)
    return f


def make_conv2d(shapeX, shapeF, tgt, tgt_host, func_name, dtype="float32"):
    assert (shapeX[1] == shapeF[1])
    N, C, H, W = shapeX  # batch, channel, i, j
    M, C, R, S = shapeF

    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum"""
    """Hint: go by conv2d definition. Treat stride=1, padding=0 case only."""
    """For a challenge, treat the general case for stride and padding."""
    A = tvm.placeholder(shapeX, dtype=dtype, name="A")  # input
    B = tvm.placeholder(shapeF, dtype=dtype, name="B")  # filter

    channel = tvm.reduce_axis((0, C), "channel")
    di = tvm.reduce_axis((0, R), "di")
    dj = tvm.reduce_axis((0, S), "dj")

    Output = tvm.compute(
        (N, M, H - R + 1, W - S + 1),
        lambda n, m, i, j: tvm.sum(A[n, channel, i + di, j + dj] * B[m, channel, di, dj], axis=[channel, di, dj]),
        name="Output")
    s = tvm.create_schedule(Output.op)
    # print (tvm.lower(s, [A, B, Output], simple_mode=True))
    f = tvm.build(s, [A, B, Output], tgt, target_host=tgt_host, name=func_name)
    return f


def make_matrix_softmax(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum, tvm.max, tvm.exp"""
    """Hint: do not reuse the same reduction axis j."""
    """Hint: implement the following version for better stability
        e_x = np.exp(x - np.max(x))
        softmax(x)= e_x / e_x.sum()
    """
    # b = y - np.max(y, axis=1, keepdims=True)
    # expb = np.exp(b)
    # softmax = expb / np.sum(expb, axis=1, keepdims=True)
    A = tvm.placeholder(shape, dtype=dtype, name="A")  # input

    classes = tvm.reduce_axis((0, shape[1]), "classes")
    classes_2 = tvm.reduce_axis((0, shape[1]), "classes_2")

    max_res = tvm.compute((shape[0],), lambda x: tvm.max(A[x, classes], axis=classes),
                          name="max_res")
    normalized_res = tvm.compute(shape, lambda x, y: A[x, y] - max_res[x],
                                 name="normalized_res")
    exp_res = tvm.compute(shape, lambda *i: tvm.exp(normalized_res(*i)), name="exp_res")
    exp_sum = tvm.compute((shape[0],), lambda x: tvm.sum(exp_res[x, classes_2], axis=classes_2),
                          name="exp_sum")
    output = tvm.compute(
        shape,
        lambda x, y: exp_res[x, y] / exp_sum[x],
        name="output")
    s = tvm.create_schedule([max_res.op, normalized_res.op, exp_res.op, output.op])
    # print(tvm.lower(s, [A, output], simple_mode=True))
    f = tvm.build(s, [A, output], tgt, target_host=tgt_host, name=func_name)
    return f


def make_matrix_softmax_cross_entropy(shape, tgt, tgt_host, func_name,
                                      dtype="float32"):
    """TODO: Your code here"""
    """Hint: output shape should be (1,)"""
    A = tvm.placeholder(shape, dtype=dtype, name="A")  # input
    Label = tvm.placeholder(shape, dtype=dtype, name="Label")  # input

    classes = tvm.reduce_axis((0, shape[1]), "classes")
    classes_2 = tvm.reduce_axis((0, shape[1]), "classes_2")
    classes_3 = tvm.reduce_axis((0, shape[1]), "classes_3")
    samples = tvm.reduce_axis((0, shape[0]), "samples")

    max_res = tvm.compute((shape[0],), lambda x: tvm.max(A[x, classes], axis=classes),
                          name="max_res")
    normalized_res = tvm.compute(shape, lambda x, y: A[x, y] - max_res[x],
                                 name="normalized_res")
    exp_res = tvm.compute(shape, lambda *i: tvm.exp(normalized_res(*i)), name="exp_res")
    exp_sum = tvm.compute((shape[0],), lambda x: tvm.sum(exp_res[x, classes_2], axis=classes_2),
                          name="exp_sum")
    label_times_log_softmax = tvm.compute(
        shape,
        lambda x, y: Label[x, y] * tvm.log(exp_res[x, y] / exp_sum[x]),
        name="label_times_log_softmax")

    summed_error = tvm.compute((shape[0],), lambda x: tvm.sum(label_times_log_softmax[x, classes_3], axis=classes_3),
                               name="summed_error")
    summed_summed_error = tvm.compute((1,), lambda _: tvm.sum(summed_error[samples], axis=samples),
                                      name="summed_summed_error ")
    output = tvm.compute((1,), lambda i: tvm.const(-1, dtype) * summed_summed_error[i] / tvm.const(shape[0], dtype),
                         name="output")
    s = tvm.create_schedule([max_res.op, normalized_res.op, exp_res.op, exp_sum.op, label_times_log_softmax.op,
                             summed_error.op, summed_summed_error.op, output.op])
    # print(tvm.lower(s, [A, Label, output], simple_mode=True))
    f = tvm.build(s, [A, Label, output], tgt, target_host=tgt_host, name=func_name)
    return f


def make_reduce_sum_axis_zero(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = topi.sum(A, axis=0, keepdims=False)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_broadcast_to(shape, to_shape, tgt, tgt_host, func_name,
                      dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = topi.broadcast_to(A, to_shape)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_sgd_update(shape, learning_rate, tgt, tgt_host, func_name,
                    dtype="float32"):
    X = tvm.placeholder(shape, dtype=dtype, name="A")
    grad = tvm.placeholder(shape, dtype=dtype, name="grad")
    Y = tvm.compute(shape, lambda *i: X(*i) - learning_rate * grad(*i))

    s = tvm.create_schedule(Y.op)
    f = tvm.build(s, [X, grad, Y], tgt, target_host=tgt_host, name=func_name)
    return f
