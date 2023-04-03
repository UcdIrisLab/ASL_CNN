"""
Create by lzw @ 20170113
Modified by Donghoon Kim at UC Davis 03212023
"""

import numpy as np
import os
import time
#import matplotlib.pyplot as plt
from scipy.io import savemat


from tensorflow.python.keras.optimizers import SGD, Adam
from tensorflow.python.keras.models import save_model, load_model
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping


from utils import save_nii_image, calc_RMSE, loss_func, loss_func_y, repack_pred_label, \
                  MRIModel, parser, xls_append_data, load_nii_image, unmask_nii_data, loss_funcs, \
                  fetch_PCASL_test_data_DK, fetch_PCASL_train_data_DK

types = ['ATT','CBF']


ntypes = len(types)
decay = 0.1

# Get parameter frome command-line input
# Definition of the command-line arguments are in model.py or displayed by `python MRI-net.py -h`
args = parser().parse_args()
mtype = args.model
train = args.train
lr = 0.001
epochs = args.epoch
batch_size = 256
nPWI = args.PWI      ##__DK       PWI = 150  what is this  \\hard input 109  DK

train_subjects = args.train_subjects
test_subject = args.test_subject[0]

combine = None  ##__DK    combine of bvals
convs = args.convs
losss = 0
con = args.con
segment = args.segment
kernels = None
tgt = 0

patch_size = 3
label_size = patch_size - 4
base = args.base

layer = 3

# Changeable parameters
shuffle=False
save_kernels = False
save_intermediate = False

movefile = None

savename = str(nPWI) + '-' + args.savename + '-' + args.model + '-' + \
           'patch' + '_' + str(patch_size) + \
           '-base_' + str(base)

print(kernels) #DK

if kernels is not None:
    savename += '-kernsle' + str(kernels)

print(savename) #DK

y_acc = None
output_acc = None
y_loss = None
output_loss = None
nepoch = None

# Define the adam optimizer
adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

# Train on the training data.
if train:
    # Define the model.
    model = MRIModel(nPWI, model=mtype, layer=layer, con=con, train=train, segment=segment, kernels=kernels)

    model.model(adam, loss_funcs[losss], patch_size, convs=convs)

    data, label = fetch_PCASL_train_data_DK(train_subjects, nPWI, mtype,
                                       patch_size=patch_size,
                                       label_size=label_size,
                                       base=base,
                                       combine=combine,
                                       segment=segment)
    print("in Train after fetch_PCASL_test_data_DK")
    print(data.shape)  #DK
    print(label.shape) #DK
    print("--------\n")

    if mtype == 'conv2d_single':
        label = label[..., tgt:tgt+1]

    if convs:
        data = np.expand_dims(data, axis=2)
        print("in convs if state  data shape is:",data)
        print("in convs if state  data shape is:",data.shape)
    print("after if conv data:",data.shape)
    reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.5, patience=10, epsilon=0.0001)
    tensorboard = TensorBoard(histogram_freq=0)
    early_stop = EarlyStopping(monitor='val_loss', patience=30, min_delta=0.0000005)
    #[nepoch, output_loss, y_loss, output_acc, y_acc]

    [nepoch, output_loss, y_loss, output_acc, y_acc] = model.train(data, label, batch_size, epochs,
                                                                   [reduce_lr, tensorboard, early_stop],
                                                                   savename, shuffle=not shuffle,
                                                                   validation_data=None)
    print("--------")
    print(output_loss)
    print("--------\n")




    print("nepoch -------",nepoch)
print("DK check point MRI-gongthing line 148-----------------------------------")
# Define the model
mask = load_nii_image('supports/mask_' + test_subject + '.nii')
tdata, tlabel = fetch_PCASL_test_data_DK(test_subject, mask, nPWI, mtype, combine=combine, segment=segment)
test_shape = None
if test_shape is None:
  test_shape = tdata.shape[1:4]
print("test shape is ",test_shape)  #DK
print("test shape is \n",test_shape)  #DK
model = MRIModel(nPWI, model=mtype, layer=layer, con=con, train=False, segment=segment, kernels=kernels, test_shape=test_shape)
model.model(adam, loss_func, segment, convs=convs)
model.load_weight(savename)

weights = model._model.layers[1].get_weights()
if save_kernels:
    savemat('conv_kernels.mat', {'data': weights[0]})

# Predict on the test data.
print("printding tdata shape and kernels")
print(tdata.shape, kernels)  #DK

time1 = time.time()
pred = model.predict(tdata)
time2 = time.time()

if save_intermediate:
    for i in range(3):
        output = np.zeros(mask.shape + (150,))
        output[1:-1, 1:-1, :, :] = pred[i].transpose(1, 2, 0, 3)
        output[mask == 0] = 0
        savemat('conv_output_' + str(i) + '.mat', {'data': output})


pred = np.concatenate((pred[0], pred[1]), axis=-1)

time3 = time.time()
pred = repack_pred_label(pred, mask, mtype, segment=segment)
time4 = time.time()

print(pred.shape)  #DK

print("predict done", time2 - time1, time4 - time3) #DK


RMSE = np.zeros(ntypes)

os.system("mkdir -p nii diff")

if mtype == 'conv2d_single':
    label = tlabel[..., tgt]
    data = pred[..., 0]
    print(data.shape) #DK
    print(label.shape) #DK
    RMSE[tgt] = calc_RMSE(data, label, mask)
    filename = 'nii/' + test_subject + '-' + types[tgt] + '-' + savename + '.nii'
    diffname = 'diff/' + test_subject + '-' + types[tgt] + '-' + savename + '.nii'

    data[mask == 0] = 0
    save_nii_image(filename, data, 'supports/mask_' + test_subject + '.nii')
    save_nii_image(diffname, data - label, 'supports/mask_' + test_subject + '.nii')
else:
    rmse = calc_RMSE(pred, tlabel, mask, model='conv3d_staged')

    for i in range(ntypes): #range(ntypes):
        data = pred[..., i]
        label = tlabel[..., i]

        RMSE[i] = calc_RMSE(data, label, mask, model='conv3d_staged')




        filename = 'nii/' + test_subject + '-' + types[i] + '-' + savename + '.nii'
        diffname = 'diff/' + test_subject + '-' + types[i] + '-' + savename + '.nii'

        data[mask == 0] = 0
        save_nii_image(filename, data, 'supports/mask_' + test_subject + '.nii', None)
        save_nii_image(diffname, data - label, 'supports/mask_' + test_subject + '.nii', None)

    RMSE_NEW = np.zeros(ntypes)


print("RMSE: ATT: ",RMSE[0],"RMSE: CBF: ",RMSE[1])  #DK

data = []
data.append(['Net', savename])
data.append(['PWI', nPWI])
data.append(['epoch', epochs])
data.append(['train sets', train_subjects])
data.append(['epoch', nepoch])
data.append(['', ''])
data.append(['test set', test_subject])
for i in range(ntypes):
    data.append([types[i], RMSE[i]])
data.append(['RMSE', rmse])

xls_append_data('RESULTS.xls', data)
