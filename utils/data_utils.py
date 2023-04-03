"""
Create @ 20170403 by lzw
Modified by Donghoon Kim at UC Davis 03212023
"""

import os
import numpy as np
from scipy.io import loadmat, savemat
from utils.nii_utils import load_nii_image, save_nii_image, mask_nii_data

def gen_PCASL_base_datasets_DK(path, subject, fdata=True, flabel=True):
    """
    Generate Datasets.
    """
    os.system("mkdir -p datasets/datas datasets/labels supports")
    ltype = ['ATTgm','CBFgm']


    print("Generating for " + subject + " Data")

    if fdata:
        data = load_nii_image(path + '/' + subject + '/PWI_timing.nii')

        savemat('datasets/datas/' + subject + '.mat', {'data':data})

    if flabel:
        mask = load_nii_image(path + '/' + subject + '/brain_mask.nii')
        os.system('cp ' +  path + '/' + subject + '/brain_mask.nii supports/mask_' + subject + '.nii')
        label = np.zeros(mask.shape + (2,)) #DK number of out
        for i in range(2): #DK number of out
            filename = path + '/' + subject + '/' + subject + '_' + ltype[i] + '.nii'
            label[:, :, :, i] = load_nii_image(filename)
        savemat('datasets/labels/' + subject + '.mat', {'label':label})


def gen_3d_patches(data, mask, size, stride):

    print(data.shape, mask.shape)
    patches = []
    for layer in np.arange(0, mask.shape[2], stride):
        for x in np.arange(0, mask.shape[0], stride):
            for y in np.arange(0, mask.shape[1], stride):
                xend, yend, layerend = np.array([x, y, layer]) + size

                lxend, lyend, llayerend = np.array([x, y, layer]) + stride
                if mask[x:lxend, y:lyend, layer:llayerend].sum() > 0:

                    patches.append(data[x:xend, y:yend, layer: layerend, :])
    return np.array(patches)

def gen_conv3d_PCASL_datasets_DK(path, subjects, patch_size, label_size, base=1, test=False):
    """
    Generate Conv3D Datasets.
    """
    print("patch size is",patch_size)
    print("label size is",label_size)
    print("base is",base)
    offset = base - (patch_size - label_size) / 2
    print("offset is",offset)
    for subject in subjects:

        print("Generating for " + subject + " Conv3D Datasets") #print "Generating for " + subject + " Conv3D Datasets" DK

        labels = loadmat('datasets/labels/' + subject+ '.mat')['label']
        labels = labels[base:-base, base:-base, base:-base, :]
        mask = load_nii_image(path + '/' + subject + '/brain_mask.nii')  #DK
        mask = mask[base:-base, base:-base, base:-base]

        data = loadmat('datasets/datas/' + subject + '.mat')['data']
        # data = data[base:-base, base:-base, base:-base, :]

        if offset:
            data = data[offset:-offset, offset:-offset, offset:-offset, :12]

        patches = gen_3d_patches(data, mask, (3,3,3), label_size) #DK_patches = gen_3d_patches(data, mask, patch_size, label_size)
        patches = patches.reshape(patches.shape[0], -1)
        savemat('datasets/datas/' + subject + '-base' + str(base) + '-patches-3d-' + str(patch_size)\
            + '-' + str(label_size) + '-all.mat', {'data':patches},  format='4')

        labels = gen_3d_patches(labels, mask, label_size, label_size)
        savemat('datasets/labels/' + subject + '-base' + str(base) + '-labels-3d-' + str(patch_size)\
                + '-' + str(label_size) + '-all.mat', {'label':labels})

        print(patches.shape)  #print patches.shape DK
        print(labels.shape)  #DK
