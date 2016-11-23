import tensorflow as tf
import numpy as np
import os
import net_tools as net
import mnist_utils as utils
from tensorflow.examples.tutorials.mnist import input_data


# hyper-parameters
learning_rate = 1e-4
use_relu = True
batch_size = 128
max_epoch = 30
train_log_dir = r'/home/longc/code/neural_project1/train_log'
ckpt_dir = r'/home/longc/code/neural_project1/train_log/model_mlp_relu.ckpt'
npz_dir = r'/home/longc/code/neural_project1/train_log/model_mlp_relu.npz'
layer_sizes = [64, 128, 10]
layer_num = len(layer_sizes)

# load minist dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, validation_size=0)
x_test = mnist.test.images
y_test = mnist.test.labels
y_test = np.argmax(y_test, axis=1)

# load npz
if not os.path.isfile(npz_dir):
    sess = tf.InteractiveSession()

    # inputs
    x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')

    # forward pass
    fc = net.fc_layer('fc1', x, layer_sizes[0], relu=use_relu)
    for i in range(1, layer_num-1):
        fc = net.fc_layer('fc{}'.format(i+1), fc, layer_sizes[i], relu=use_relu)

    logits = net.fc_layer('fc{}'.format(layer_num), fc, layer_sizes[-1], linear=True)
    y = tf.nn.softmax(logits, name='y')

    # init
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_dir)

    # save npz
    var_list = tf.all_variables()
    params = utils.save_npz(var_list, sess, name=npz_dir)
else:
    params = utils.load_npz(npz_dir)
assert(len(params) == 2*layer_num)

# leaky integrate-and-fire (LIF) neurons
# initialize network
num_examples = len(x_test)
mems = list()
sum_spikes = list()
for i, size in enumerate(layer_sizes):
    mems.append(np.zeros([num_examples, size], dtype=np.float32))
    sum_spikes.append(np.zeros([num_examples, size], dtype=np.float32))

dt = 0.001
duration = 0.05
threshold = 1
rest = 0.
max_rate = 600

for t in np.arange(dt, duration+dt, dt):
    # Create poisson distributed spikes from the input images
    rescale_fac = 1. / (dt * max_rate)
    spike_snapshot = np.random.rand(*x_test.shape) * rescale_fac
    spikes = np.asarray(spike_snapshot <= x_test, dtype=np.float32)
    for i in range(layer_num):
        W = params[i*2]
        b = params[i*2+1]
        # Get input impulse from incoming spikes
        impulse = np.matmul(spikes, W) + b
        # Add input to membrane potential
        mems[i] += impulse
        # Check for spiking and Rest
        spikes = np.asarray(mems[i] > np.array(threshold), dtype=np.float32)
        mems[i][spikes > 0] = 0
        # Store result for analysis later
        sum_spikes[i] += spikes

    predict = np.argmax(sum_spikes[-1], axis=1)
    correct = np.sum(predict == y_test, dtype=np.float32)
    acc = correct / num_examples * 100
    print('Time: %1.3fs | Accuracy: %2.2f%%' % (t, acc))
