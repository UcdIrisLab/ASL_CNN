"""
Create by lzw @ 20170117
Modified by Donghoon Kim at UC Davis 03212023
"""

import os
import numpy as np
from scipy.io import loadmat


def repack_pred_label(pred, mask, model, segment=False):
    """
    Get.
    """
    if segment:
        label = np.zeros(mask.shape + (9,))
    elif model == 'conv2d_single':
        label = np.zeros(mask.shape + (1,))
    else:
        label = np.zeros(mask.shape + (2,)) #label = np.zeros(mask.shape + (8,)) #DK number of out

    if model[:6] == 'conv2d':
        label[1:-1, 1:-1, :, :] = pred.transpose(1, 2, 0, 3)
    elif model[:6] == 'conv3d':
        label[1:-1, 1:-1, 1:-1, :] = pred  #DK label[1:-1, 1:-1, 1:-1, :] = pred
    else:
        label = pred.reshape(label.shape)

    return label

# DK

def fetch_PCASL_train_data_DK(subjects, nPWI, model, patch_size=3, label_size=1, base=1,
                     whiten=True, combine=None, segment=False):
    """
    #Fetch train data.
    """
    datas = None
    labels = None

    if model[0:6] == 'conv2d':
        filename = '-2d-' + str(patch_size) + '-' + str(label_size)
    elif model[:6] == 'conv3d':
        filename = '-3d-' + str(patch_size) + str(label_size)
    else:
        filename = '-1d'

    for subject in subjects:
        lpath = 'datasets/labels/' + subject
        label = loadmat(lpath + '-base' + str(base) + '-labels' + filename + '-all.mat')['label']

        dpath = 'datasets/datas/' + subject
        data = loadmat(dpath + '-base' + str(base) + '-patches' + filename + '-all.mat')['data']
        #data = loadmat(dpath + '-base' + str(base) + '-patches' + filename + '-averaged-all.mat')['data']

        for i in range(label.shape[0]):
            if np.isnan(label[i]).any():
                label[i] = 0
                data[i] = 0

        if datas is None:
            datas = data
            labels = label
        else:
            datas = np.concatenate((datas, data), axis=0)
            labels = np.concatenate((labels, label), axis=0)

    data = np.array(datas)
    label = np.array(labels)

    if model[:6] == 'conv3d':
        data = data.reshape(data.shape[0], 3, 3, 3, nPWI) #DK data = data.reshape(data.shape[0], 3, 3, 3, nPWI)

    # Select the inputs.
    if combine is not None:
        data = data[..., combine == 0]
    else:
        data = data[..., :nPWI]
    #print data.mean()  DK
    print(data.mean())

    # whiten the data.
    if whiten:
        data = data / data.mean() - 1.0

    if model == 'conv0d':
        data = np.expand_dims(data, axis=-2)
        data = np.expand_dims(data, axis=-2)
        data = np.expand_dims(data, axis=-2)

    return data, label

def fetch_PCASL_test_data_DK(subject, mask, nPWI, model, patch_size=3, label_size=1, base=1,
                    whiten=True, combine=None, segment=False):
    """
    Fetch test data.
    """

    label = loadmat('datasets/labels/' + subject + '.mat')['label']

    data = loadmat('datasets/datas/' + subject + '.mat')['data']

    for i in range(label.shape[0]):
        if np.isnan(label[i]).any():
            label[i] = 0
    # data[i] = 0

    #print data.shape DK
    #print label.shape DK
    print(data.shape)
    print(label.shape)

    # Select the inputs.
    if combine is not None:
        data = data[..., combine == 0]
    else:
        data = data[..., :nPWI]

    data.mean()

    # Whiten the data.
    if whiten:
        data = data / data[mask > 0].mean() - 1.0

    # ReShape the data to suit the model.
    if model[:6] == 'conv3d':
        data = np.expand_dims(data, axis=0)
    elif model[:6] == 'conv2d':
        data = data.transpose((2, 0, 1, 3))
    elif model == 'conv0d':
        data = data
    else:
        data = data.reshape(-1, data.shape[-1])

    return data, label
