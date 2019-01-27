import numpy as np
import _pickle as cPickle

IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_DEPTH = 3
NUM_CLASS = 10

full_data_path = 'cifar10_data/data_batch_'
vali_path = 'cifar10_data/test_batch'


CIFAR10_LABELS_LIST = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

def _read_one_batch(path):

    fo = open(path, 'rb')
    #dicts = cPickle.load(fo)
    dicts = cPickle.load(fo, encoding='latin1')
    fo.close()

    data = dicts['data']
    label = np.array(dicts['labels'])

    return data, label

def read_training_data():
    '''
    Read in validation data. Whitening at the same time
    :return: Validation image data as 4D numpy array. Validation labels as 1D numpy array
    '''

    data = np.array([]).reshape([0, IMG_WIDTH * IMG_HEIGHT * IMG_DEPTH])
    label = np.array([]).reshape([0,10])

    path_list = []
    for i in range(1, 6): # num of batches is 1~5
        path_list.append(full_data_path + str(i))

    for address in path_list:
        print('Reading images from ' + address)
        batch_data, batch_label = _read_one_batch(address)
        # Concatenate along axis 0 by default
        data = np.concatenate((data, batch_data))

        encoded = np.zeros((len(batch_label), 10))

        for idx, val in enumerate(batch_label):
            encoded[idx][val] = 1

        label = np.concatenate((label, encoded))

    num_data = len(label)

    data = data.reshape((num_data, IMG_HEIGHT * IMG_WIDTH, IMG_DEPTH), order='F')
    data = data.reshape((num_data, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))

    data = data.astype(np.float32)

    return data, label


def load_cfar10_batch(batch_id): #for train data

    batch_data, batch_label = _read_one_batch(full_data_path + str(batch_id))
    features = batch_data.reshape((len(batch_data), 3, 32, 32)).transpose(0, 2, 3, 1)

    return features, batch_label


def read_validation_data():
    '''
    Read in validation data. Whitening at the same time
    :return: Validation image data as 4D numpy array. Validation labels as 1D numpy array
    '''

    b_data, label = _read_one_batch(vali_path)

    data = np.array([]).reshape([0, IMG_WIDTH * IMG_HEIGHT * IMG_DEPTH])
    data = np.concatenate((data, b_data))
    encoded = np.zeros((len(label), 10))

    for idx, val in enumerate(label):
        encoded[idx][val] = 1

    num_data = len(label)

    data = data.reshape((num_data, IMG_HEIGHT * IMG_WIDTH, IMG_DEPTH), order='F')
    data = data.reshape((num_data, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))

    data = data.astype(np.float32)

    return data, encoded
