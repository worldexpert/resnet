import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

from cifar10_input import *

BN_EPSILON = 0.001



def activation_summary(x):
    '''
    :param x: A Tensor
    :return: Add histogram summary and scalar summary of the sparsity of the tensor
    '''
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def batch_normalization_layer(name, input_layer, dimension, is_training=True):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        bn_layer = tf.identity(input_layer)

    return bn_layer

def residual_block(input_layer, output_channel, is_training, first_block=False):
    '''
    Defines a residual block in ResNet
    :param input_layer: 4D tensor
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :param first_block: if this is the first residual block of the whole network
    :return: 4D tensor.
    '''
    input_channel = input_layer.get_shape().as_list()[-1]

    # When it's time to "shrink" the image size, we use stride = 2
    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    # About : https://timodenk.com/blog/tensorflow-batch-normalization/
    # The first conv layer of the first residual block does not need to be normalized and relu-ed.
    with tf.variable_scope('conv1_in_block'):
        if first_block:
            conv1 = tf.layers.conv2d(input_layer, output_channel, [3, 3], activation=tf.nn.relu, padding='SAME')

        else:
            bn_layer= tf.layers.batch_normalization(input_layer, training= is_training)
            #bn_layer = batch_normalization_layer(input_layer, input_layer.get_shape().as_list()[-1])
            conv1 = tf.layers.conv2d(bn_layer, output_channel, [3, 3], activation=tf.nn.relu, padding='SAME', strides=(stride, stride))


    with tf.variable_scope('conv2_in_block'):
        bn_layer= tf.layers.batch_normalization(conv1, training= is_training)
        #bn_layer = batch_normalization_layer(conv1, conv1.get_shape().as_list()[-1])
        conv2 = tf.layers.conv2d(bn_layer, output_channel, [3, 3], activation=tf.nn.relu, padding='SAME')


    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    #  depth of input layers
    if increase_dim is True:

        pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='VALID')
        padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                      input_channel // 2]])

    else:
        padded_input = input_layer

    output = conv2 + padded_input
    return output

def whitening_image(image_np):
    '''
    Performs per_image_whitening
    :param image_np: a 4D numpy array representing a batch of images
    :return: the image numpy array after whitened
    '''
    tmp = np.empty(image_np.shape)
    for i in range(len(image_np)):
        mean = np.mean(image_np[i, ...])
        # Use adjusted standard deviation here, in case the std == 0.
        std = np.max([np.std(image_np[i, ...]), 1.0/np.sqrt(IMG_HEIGHT * IMG_WIDTH * IMG_DEPTH)])
        tmp[i,...] = (image_np[i, ...] - mean) / std
    return tmp


def random_crop_and_flip(batch_data, padding_size):
    '''
    Helper to random crop and random flip a batch of images
    :param padding_size: int. how many layers of 0 padding was added to each side
    :param batch_data: a 4D batch array
    :return: randomly cropped and flipped image
    '''
    cropped_batch = np.zeros(len(batch_data) * IMG_HEIGHT * IMG_WIDTH * IMG_DEPTH).reshape(
        len(batch_data), IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH)

    pad_width = ((0, 0), (padding_size, padding_size), (padding_size, padding_size), (0, 0))
    batch_data = np.pad(batch_data, pad_width=pad_width, mode='constant', constant_values=0)

    for i in range(len(batch_data)):
        x_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        y_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        cropped_batch[i, ...] = batch_data[i, ...][x_offset:x_offset+IMG_HEIGHT,
                      y_offset:y_offset+IMG_WIDTH, :]

        flip_prop = np.random.randint(low=0, high=2)
        if flip_prop == 0:
            cropped_batch[i, ...] = cv2.flip(cropped_batch[i, ...], 1) #axis=1

    return cropped_batch

def generate_augment_train_batch(train_data, train_labels, train_batch_size):
    '''
    This function helps generate a batch of train data, and random crop, horizontally flip
    and whiten them at the same time
    :param train_data: 4D numpy array
    :param train_labels: 1D numpy array
    :param train_batch_size: int
    :return: augmented train batch data and labels. 4D numpy array and 1D numpy array
    '''
    EPOCH_SIZE = 10000 * 5
    padding_size= 2

    offset = np.random.choice(EPOCH_SIZE - train_batch_size, 1)[0]
    batch_data = train_data[offset:offset+train_batch_size, ...]
    batch_data = random_crop_and_flip(batch_data, padding_size=padding_size)

    batch_data = whitening_image(batch_data)
    batch_label = train_labels[offset:offset+train_batch_size, ...]

    return batch_data, batch_label

def generate_vali_batch(vali_data, vali_label, vali_batch_size):
    '''
    If you want to use a random batch of validation data to validate instead of using the
    whole validation data, this function helps you generate that batch
    :param vali_data: 4D numpy array
    :param vali_label: 1D numpy array
    :param vali_batch_size: int
    :return: 4D numpy array and 1D numpy array
    '''
    offset = np.random.choice(10000 - vali_batch_size, 1)[0]
    vali_data_batch = vali_data[offset:offset+vali_batch_size, ...]
    vali_label_batch = vali_label[offset:offset+vali_batch_size]
    return vali_data_batch, vali_label_batch

def display_random_image(image, labels) :
    fig = plt.figure()
    order = 1
    for index in np.random.choice(10000, 10):

        if order > 10:
            break
        subplot = fig.add_subplot(2, 5, order)
        subplot.set_xticks([])
        subplot.set_yticks([])

        # subplot.set_title("%d %d" % (np.argmax(labels[i]), np.argmax(test_labels[i])))
        subplot.set_title('{}: {}'.format(np.argmax(labels[index]), CIFAR10_LABELS_LIST[np.argmax(labels[index])]))

        data = image[index].reshape((32 * 32, 3))
        data = np.transpose(data)
        im_r = data[0].reshape((32, 32)) / 255.0
        im_g = data[1].reshape((32, 32)) / 255.0
        im_b = data[2].reshape((32, 32)) / 255.0

        img = np.dstack((im_r, im_g, im_b))

        subplot.imshow(img)
        order += 1

    plt.show()