import tensorflow as tf
import numpy as np
import os
import scipy.misc as misc
from PIL import Image
def read_crop_data(path, batch_size, shape, factor):
    h = shape[0]
    w = shape[1]
    c = shape[2]
    filenames = os.listdir(path)
    rand_selects = np.random.randint(0, filenames.__len__(), [batch_size])
    batch = np.zeros([batch_size, h, w, c])
    downsampled = np.zeros([batch_size, h//factor, w//factor, c])
    for idx, select in enumerate(rand_selects):
        try:
            img = np.array(Image.open(path + filenames[select]))[:, :, :3]
            crop = random_crop(img, h)
            batch[idx, :, :, :] = crop
            downsampled[idx, :, :, :] = misc.imresize(crop, [h // factor, w // factor])
        except:
            img = np.array(Image.open(path + filenames[0]))[:, :, :3]
            crop = random_crop(img, h)
            batch[idx, :, :, :] = crop
            downsampled[idx, :, :, :] = misc.imresize(crop, [h//factor, w//factor])
    return batch/127.5-1, downsampled/127.5-1

def random_crop(img, size):
    h = img.shape[0]
    w = img.shape[1]
    start_x = np.random.randint(0, h - size + 1)
    start_y = np.random.randint(0, w - size + 1)
    return img[start_x:start_x + size, start_y:start_y + size, :]
def conv(name, inputs, nums_out, k_size, strides=1, is_SN=False):
    nums_in = int(inputs.shape[-1])
    with tf.variable_scope(name):
        kernel = tf.get_variable(name+"weights", [k_size, k_size, nums_in, nums_out], initializer=tf.truncated_normal_initializer(stddev=0.02))
        bias = tf.get_variable(name+"bias", [nums_out], initializer=tf.constant_initializer(0.))
        if is_SN:
            inputs = tf.nn.conv2d(inputs, spectral_normalization(name, kernel), [1, strides, strides, 1], "SAME")
            inputs=tf.nn.bias_add(inputs,bias)
        else:
            inputs = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], "SAME")
            inputs=tf.nn.bias_add(inputs,bias)
    return inputs

def atrous_conv(name,inputs,nums_out,k_size,rate):
    nums_in=int(inputs.shape[-1])
    with tf.variable_scope(name):
        kernel = tf.get_variable(name+"a_weight",[k_size,k_size,nums_in,nums_out],initializer=tf.truncated_normal_initializer(stddev=0.02))
        bias = tf.get_variable(name+"bias",[nums_out],initializer=tf.constant_initializer(0.))
        input = tf.nn.atrous_conv2d(inputs,kernel,rate=rate,padding="SAME")
        input = tf.nn.bias_add(input,bias)
    return input

def conv_(inputs, w, b):
    inputs=tf.nn.conv2d(inputs, w, [1, 1, 1, 1], "SAME")
    inputs=tf.nn.bias_add(inputs,b)
    return inputs

def max_pooling(inputs):
    return tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

def _l2normalize(v, eps=1e-12):
    return v / tf.sqrt(tf.reduce_sum(tf.square(v)) + eps)

def max_singular_value(W, u=None, Ip=1):
    if u is None:
        u = tf.get_variable("u", [1, W.shape[-1]], initializer=tf.random_normal_initializer(), trainable=False) #1 x ch
    _u = u
    _v = 0
    for _ in range(Ip):
        _v = _l2normalize(tf.matmul(_u, W), eps=1e-12)
        _u = _l2normalize(tf.matmul(_v, W, transpose_b=True), eps=1e-12)
    sigma = tf.reduce_sum(tf.matmul(_u, W) * _v)
    return sigma, _u, _v

def spectral_normalization(name, W, Ip=1):
    u = tf.get_variable(name + "_u", [1, W.shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)  # 1 x ch
    W_mat = tf.transpose(tf.reshape(W, [-1, W.shape[-1]]))
    sigma, _u, _ = max_singular_value(W_mat, u, Ip)
    with tf.control_dependencies([tf.assign(u, _u)]):
        W_sn = W / sigma
    return W_sn
def prelu(name, inputs ,trainable=True):
    with tf.variable_scope(name):
        slope = tf.get_variable(name+"alpha", [1], initializer=tf.constant_initializer(0.01),trainable=trainable)
    return tf.maximum(inputs, inputs * slope)

def leaky_relu(inputs, slope=0.2):
    return tf.maximum(slope * inputs, inputs)

def phaseShift(inputs, scale, shape_1, shape_2):
    X = tf.reshape(inputs, shape_1)
    X = tf.transpose(X, [0, 1, 4, 2, 3])
    return tf.reshape(X, shape_2)

def pixelShuffler(inputs, scale=2):
    size = tf.shape(inputs)
    batch_size = size[0]
    h = size[1]
    w = size[2]
    c = inputs.get_shape().as_list()[-1]
    # Get the target channel size
    channel_target = c // (scale * scale)
    channel_factor = c // channel_target
    shape_1 = [batch_size, h, w, channel_factor // scale, channel_factor // scale]
    shape_2 = [batch_size, h * scale, w * scale, 1]
    # Reshape and transpose for periodic shuffling for each channel
    input_split = tf.split(inputs, channel_target, axis=3)
    output = tf.concat([phaseShift(x, scale, shape_1, shape_2) for x in input_split], axis=3)
    return output

def RDB(name, inputs,inputs_, C_nums, input_num, out_num):
    with tf.variable_scope("RDB_"+name):
        for i in range(C_nums):
            x1 = leaky_relu(conv("conv2_"+ str(i) , inputs,input_num, 3,  1))
            x2 = tf.concat([inputs, x1], axis=-1)
            x3 = leaky_relu(atrous_conv("conv3_"+ str(i), x2, out_num, 1, 2))
            inputs = tf.concat([inputs, x3], axis=-1)
        inputs = conv("conv",inputs,out_num,1,1)
    return inputs
def RDB_(name, inputs, C_nums, input_num, out_num):
    with tf.variable_scope("RDB_"+name):
        temp = tf.identity(inputs)
        for i in range(C_nums):
            x = leaky_relu(conv("conv2_" + str(i), inputs,input_num, 3,  1))
            inputs = tf.concat([inputs, x], axis=-1)
        inputs = conv("conv", inputs,  out_num,1, 1)
    return inputs

def batchnorm(x, train_phase, scope_bn):
    #Batch Normalization
    #Ioffe S, Szegedy C. Batch normalization: accelerating deep network training by reducing internal covariate shift[J]. 2015:448-456.
    with tf.variable_scope(scope_bn):
        beta = tf.Variable(tf.constant(0.0, shape=[x.shape[-1]]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)
        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(train_phase, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed
def fully_connected(name, inputs, nums_out, is_SN=False):
    inputs = tf.layers.flatten(inputs)
    with tf.variable_scope(name):
        W = tf.get_variable("weights", [inputs.shape[-1], nums_out], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable("bias", [nums_out])
    if is_SN:
        return tf.matmul(inputs, spectral_normalization(name, W)) + b
    else:
        return tf.matmul(inputs, W) + b
def relu(inputs):
    return tf.nn.relu(inputs)

def avg_pool(inputs, k_size=3, strides=2, padding="SAME"):
    return tf.nn.avg_pool(inputs, [1, k_size, k_size, 1], [1, strides, strides, 1], padding)

def DownBlock(name, inputs, k_size, nums_out, is_down=True):
    #inputs: B x H x W x C_in
    with tf.variable_scope(name):
        temp = inputs
        inputs = relu(inputs)
        inputs = conv("conv1", inputs,nums_out,k_size, 1, True)  # inputs: B x H/2 x W/2 x C_out
        inputs = relu(inputs)
        inputs = conv("conv2", inputs,nums_out, k_size,  1, True)  # inputs: B x H/2 x W/2 x C_out
        if is_down:
            inputs = avg_pool(inputs)
            down_sampling = conv("down_sampling_" + name, temp,nums_out, 1,  1, True)  # down_sampling: B x H x W x C_out
            down_sampling = avg_pool(down_sampling)
            outputs = inputs + down_sampling
        else:
            outputs = inputs + temp
    return outputs

def Linear(name, inputs, nums_in, nums_out, is_sn=True):
    W = tf.get_variable("W_" + name, [nums_in, nums_out], initializer=tf.truncated_normal_initializer(stddev=0.02))
    b = tf.get_variable("B_" + name, [nums_out], initializer=tf.constant_initializer([0.]))
    if is_sn:
        return tf.matmul(inputs, spectral_normalization(name, W)) + b
    else:
        return tf.matmul(inputs, W) + b


def global_sum_pooling(inputs):
    return tf.reduce_sum(inputs, axis=[1, 2])

def Inner_product(inputs, y):
    with tf.variable_scope("IP"):
        inputs = conv("conv", inputs, 3, 3, 1, True)
    inputs = tf.reduce_sum(inputs * y,axis=[1,2,3])
    return inputs

