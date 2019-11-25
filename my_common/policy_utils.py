import tensorflow as tf


def pgn_conv(input_tensor, scope, *, n_filters, stride,
             pad='VALID', data_format='NHWC', one_dim_bias=False, ww=None, bb=None):
    """data 格式"""
    if data_format == 'NHWC':
        strides = [1, stride, stride, 1]
        bshape = [1, 1, 1, n_filters]
    elif data_format == 'NCHW':
        strides = [1, 1, stride, stride]
        bshape = [1, n_filters, 1, 1]
    else:
        raise NotImplementedError

    weight1 = tf.convert_to_tensor(ww, dtype=tf.float32)
    bias1 = tf.convert_to_tensor(bb, dtype=tf.float32)
    with tf.variable_scope(scope):
        weight = tf.get_variable("w", initializer=weight1, trainable=False)
        bias = tf.get_variable("b", initializer=bias1, trainable=False)
        if not one_dim_bias and data_format == 'NHWC':
            bias = tf.reshape(bias, bshape)
        return bias + tf.nn.conv2d(input_tensor, weight, strides=strides, padding=pad, data_format=data_format)


def pgn_linear(input_tensor, scope, *, ww=None, bb=None):
    with tf.variable_scope(scope):
        weight1 = tf.convert_to_tensor(ww, dtype=tf.float32)
        bias1 = tf.convert_to_tensor(bb, dtype=tf.float32)
        weight = tf.get_variable("w", initializer=weight1, trainable=False)
        bias = tf.get_variable("b", initializer=bias1, trainable=False)
        return tf.matmul(input_tensor, weight) + bias


def pgn_vf_linear(input_tensor, scope, n_hidden, *, ww=None, bb=None):
    with tf.variable_scope(scope):
        n_input = input_tensor.get_shape()[1].value
        weight = tf.get_variable("w", [n_input, n_hidden], initializer=tf.constant_initializer(ww))
        bias = tf.get_variable("b", [n_hidden], initializer=tf.constant_initializer(bb))
        return tf.matmul(input_tensor, weight) + bias
