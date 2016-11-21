import tensorflow as tf
import numpy as np
import os
import net_tools as net
from tensorflow.examples.tutorials.mnist import input_data


# hyper-parameters
learning_rate = 1e-4
use_relu = False
batch_size = 128
max_epoch = 30
train_log_dir = r'/home/longc/code/neural_project1/train_log'
ckpt_dir = r'/home/longc/code/neural_project1/train_log/model_mlp.ckpt-14041'
load_ckpt = False

# load minist dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, validation_size=0)

sess = tf.InteractiveSession()

# inputs
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')

# forward pass
fc1 = net.fc_layer('fc1', x, 64, relu=use_relu)
fc2 = net.fc_layer('fc2', fc1, 128, relu=use_relu)
logits = net.fc_layer('fc3', fc2, 10, linear=True)
y = tf.nn.softmax(logits, name='y')

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
correct_cnt_op = tf.reduce_sum(tf.cast(correct_prediction, tf.int32))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# backward pass
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, y_, name='cross_entropy')
loss_op = tf.reduce_mean(cross_entropy)
trainable_var = tf.all_variables()
global_step = tf.Variable(0, trainable=False)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_op, global_step=global_step, var_list=trainable_var)

# init
sess.run(tf.initialize_all_variables())
saver = tf.train.Saver()

# training
if not load_ckpt:
    step = 0
    while True:
        batch = mnist.train.next_batch(batch_size)
        if mnist.train.epochs_completed >= max_epoch:
            break

        feed_dict = {x: batch[0], y_: batch[1]}
        loss, accuracy, _ = sess.run([loss_op, accuracy_op, train_step], feed_dict=feed_dict)

        if step % 100 == 0:
            print('training: step:%d, epoch=%d, loss=%.4f, accuracy=%.4f' % (step, mnist.train.epochs_completed, loss, accuracy))
        step += 1

    #save ckpt
    checkpoint_path = os.path.join(train_log_dir, 'model_mlp.ckpt')
    saver.save(sess, checkpoint_path, global_step=step)
else:
    saver.restore(sess, ckpt_dir)
    print('load model successfully')

# testing
test_batch_size = 100
correct_cnt = 0
sample_cnt = 0
while True:
    batch = mnist.test.next_batch(test_batch_size)
    if mnist.test.epochs_completed >= 1:
        break
    feed_dict = {x: batch[0], y_: batch[1]}
    correct_cnt += sess.run(correct_cnt_op, feed_dict=feed_dict)
    sample_cnt += test_batch_size
assert(sample_cnt == 10000)

test_accuracy = float(correct_cnt) / sample_cnt
print('test: accuracy=%.4f' % (test_accuracy,))
