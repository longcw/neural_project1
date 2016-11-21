import tensorflow as tf
import numpy as np
from datetime import datetime
import os
import net_tools as net
from tensorflow.examples.tutorials.mnist import input_data


# hyper-parameters
learning_rate = 1e-4
use_relu = True
batch_size = 128
max_epoch = 60
validation_size = 5000
train_log_dir = r'/home/longc/code/neural_project1/train_log'
ckpt_dir = os.path.join(train_log_dir, r'model_cnn.ckpt')
load_ckpt = True

# load minist dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, validation_size=validation_size)

sess = tf.InteractiveSession()

# inputs
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
x_image = tf.reshape(x, [-1, 28, 28, 1], name='x_image')

# forward pass
conv1_1 = net.conv_layer('conv1_1', x_image, 32, 3, 1)
conv1_2 = net.conv_layer('conv1_2', conv1_1, 32, 3, 1)
pool1 = net.pooling_layer('pool1', conv1_2, 2, 2)

conv2_1 = net.conv_layer('conv2_1', pool1, 64, 3, 1)
conv2_2 = net.conv_layer('conv2_2', conv2_1, 64, 3, 1)
pool2 = net.pooling_layer('pool2', conv2_2, 2, 2)

fc3 = net.fc_layer('fc3', pool2, 64, relu=use_relu, flat=True, wd=0.01)
fc4 = net.fc_layer('fc4', fc3, 128, relu=use_relu, wd=0.01)
fc4_drop = tf.nn.dropout(fc4, keep_prob)
logits = net.fc_layer('fc5', fc4_drop, 10, linear=True)
y = tf.nn.softmax(logits, name='y')

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
correct_cnt_op = tf.reduce_sum(tf.cast(correct_prediction, tf.int32))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.scalar_summary('train_accuracy', accuracy_op)
tf.scalar_summary('val_accuracy', accuracy_op, collections=['validation'])

# backward pass
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, y_, name='cross_entropy')
tf.add_to_collection('losses', tf.reduce_mean(cross_entropy, name='cross_entropy_mean'))
loss_op = tf.add_n(tf.get_collection('losses'), name='total_loss')
net.add_loss_summaries(loss_op)

trainable_var = tf.all_variables()
global_step = tf.Variable(0, trainable=False)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_op, global_step=global_step, var_list=trainable_var)

# init
sess.run(tf.initialize_all_variables())
saver = tf.train.Saver(max_to_keep=20)

# training
if not load_ckpt:
    # summary
    train_summary_op = tf.merge_all_summaries()
    val_summary_op = tf.merge_all_summaries(key='validation')
    summary_dir = os.path.join(train_log_dir, datetime.now().strftime('%Y-%m-%d_%H_%M_%S'))
    summary_writer = tf.train.SummaryWriter(summary_dir, sess.graph)
    checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')

    step = 0
    while True:
        batch = mnist.train.next_batch(batch_size)
        if mnist.train.epochs_completed >= max_epoch:
            break

        feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5}
        loss, accuracy, _ = sess.run([loss_op, accuracy_op, train_step], feed_dict=feed_dict)

        if step % 50 == 0:
            print('training: step:%d, epoch=%d, loss=%.4f, accuracy=%.4f' % (step, mnist.train.epochs_completed, loss, accuracy))

        if step % 100 == 0:
            summary_str = sess.run(val_summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=step)

            # validation
            if validation_size > 0:
                val_batch = mnist.validation.next_batch(batch_size)
                feed_dict = {x: val_batch[0], y_: val_batch[1], keep_prob: 1.0}
                summary_str = sess.run(train_summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, global_step=step)
                summary_writer.flush()

        # save ckpt
        if step % 1000 == 0 and step > 0:
            saver.save(sess, checkpoint_path, global_step=step)
        step += 1
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
    feed_dict = {x: batch[0], y_: batch[1], keep_prob: 1.0}
    correct_cnt += sess.run(correct_cnt_op, feed_dict=feed_dict)
    sample_cnt += test_batch_size
assert(sample_cnt == 10000)

test_accuracy = float(correct_cnt) / sample_cnt
print('test: accuracy=%.4f' % (test_accuracy,))
