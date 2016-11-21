import tensorflow as tf
import numpy as np


def add_loss_summaries(total_loss):

    losses = tf.get_collection('losses')

    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.scalar_summary(l.op.name + ' (raw)train', l)
        tf.scalar_summary(l.op.name + ' (raw)val', l, collections=['validation'])


def activation_summary(x):
    tensor_name = x.op.name
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_with_weight_decay(wshape, stddev, trainable=True, wd=None):
    weights = tf.get_variable('weights', wshape, dtype=tf.float32,
                              initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32), trainable=trainable)
    biases = tf.get_variable('biases', [wshape[-1]], dtype=tf.float32, initializer=tf.constant_initializer(0.1), trainable=trainable)

    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(weights), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return weights, biases


def conv_layer(name, inputs, filters, size, stride, trainable=True, relu=True, reuse=False):
    with tf.variable_scope(name, reuse=reuse) as scope:
        channels = inputs.get_shape()[3]

        weight, biases = _variable_with_weight_decay([size, size, int(channels), filters], stddev=0.1, trainable=trainable)

        conv = tf.nn.conv2d(inputs, weight, strides=[1, stride, stride, 1], padding='SAME')
        conv_biased = tf.nn.bias_add(conv, biases)
        if relu:
            return tf.nn.relu(conv_biased, name=scope.name)
        else:
            return conv_biased


def conv_transpose_layer(name, inputs, filters, size, stride, trainable=True, relu=True):
    with tf.variable_scope(name) as scope:
        input_shape = inputs.get_shape()
        channels = input_shape[3]

        weight, _ = _variable_with_weight_decay([size, size, int(channels), filters], stddev=0.1, trainable=trainable)

        output_shape = np.asarray(input_shape.as_list(), dtype=np.int32)
        output_shape[1:3] *= (1 << (stride-1))
        print(output_shape)
        convt = tf.nn.conv2d_transpose(inputs, weight, output_shape=output_shape, strides=[1, stride, stride, 1], padding='SAME')
        if relu:
            return tf.nn.relu(convt, name=scope.name)
        else:
            return convt


def pooling_layer(name, inputs, size, stride):
    return tf.nn.max_pool(inputs, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME',
                          name=name)


def fc_layer(name, inputs, hiddens, flat=False, linear=False, relu=True, trainable=True, wd=None, reuse=False):
    with tf.variable_scope(name, reuse=reuse) as scope:
        input_shape = inputs.get_shape().as_list()
        if flat:
            dim = input_shape[1] * input_shape[2] * input_shape[3]
            inputs_transposed = tf.transpose(inputs, (0, 3, 1, 2))
            inputs_processed = tf.reshape(inputs_transposed, [-1, dim])
        else:
            dim = input_shape[1]
            inputs_processed = inputs

        weight, biases = _variable_with_weight_decay([dim, hiddens], stddev=1./np.sqrt(float(dim)), trainable=trainable, wd=wd)

        if linear:
            return tf.add(tf.matmul(inputs_processed, weight), biases, name=scope.name)

        ip = tf.add(tf.matmul(inputs_processed, weight), biases)
        if relu:
            output = tf.nn.relu(ip, name=scope.name)
        else:
            output = tf.nn.sigmoid(ip, name=scope.name)
        return output


def flat_layer(name, conv_outputs):
    with tf.name_scope(name):
        input_shape = conv_outputs.get_shape().as_list()
        dim = input_shape[1] * input_shape[2] * input_shape[3]
        features_transposed = tf.transpose(conv_outputs, (0, 3, 1, 2))
        features_processed = tf.reshape(features_transposed, [-1, dim])
        return features_processed


# def cosine_similarity_layer(vec_a, vec_b):
#     norm_a = tf.sqrt(tf.reduce_sum(tf.square(vec_a), 1, keep_dims=True))
#     norm_b = tf.sqrt(tf.reduce_sum(tf.square(vec_b), 1, keep_dims=True))
#     normalized_a = vec_a / norm_a
#     normalized_b = vec_b / norm_b
#     similarity = tf.matmul(normalized_a[:], normalized_b[:], transpose_b=True)
#
#     return similarity

