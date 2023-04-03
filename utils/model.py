"""
Module definition for MRI net.
Create by lzw @ 20170113
Modified by Donghoon Kim at UC Davis 03212023
"""

import os
import argparse
import numpy as np
from tensorflow.contrib.keras.api.keras.models import Model, Sequential
from tensorflow.contrib.keras.api.keras import regularizers
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Input, Conv2D, Conv3D, Flatten, Convolution1D, Reshape, Conv2DTranspose, UpSampling2D, Concatenate
class MRIModel(object):
    """
    MRI models
    """

    _nPWI = 0
    _single = False
    _model = None
    _type = ''
    _loss = []
    _label = ''
    _kernel1 = 100
    _kernel2 = 100
    _kernel3 = 100

    def __init__(self, nPWI=9, model='sequence', layer=3, con=False, train=True, segment=False, kernels=None, test_shape=[56, 70, 36]):
        self._nPWI = nPWI
        self._type = model
        self._hist = None
        self._con = con
        self._train = train
        self._seg = segment
        self._layer = layer
        self._test_shape = test_shape
        if kernels is not None:
            self._kernel1, self._kernel2, self._kernel3 = kernels



    def _conv3d_staged_model(self, patch_size, convs=None):
        """
        Conv3D model.
        """
        if self._train:
            inputs = Input(shape=(3,3,3, self._nPWI)) #DK inputs = Input(shape=(patch_size, patch_size, patch_size, self._nPWI))
        else:
            (dim0, dim1, dim2) = (self._test_shape[0], self._test_shape[1], self._test_shape[2])
            inputs = Input(shape=(dim0, dim1, dim2, self._nPWI))
        hidden = Conv3D(45, 3, strides=1, activation='relu', padding='valid')(inputs)

        watch1 = hidden
        hidden = Conv3D(45, 1, strides=1, activation='relu', padding='valid')(hidden)
        hidden = Conv3D(45, 1, strides=1, activation='relu', padding='valid')(hidden)
        watch2 = hidden
        middle = hidden
        y = Conv3D(1, 1, strides=1, activation='relu', padding='valid', name='y')(middle)
        hidden = Conv3D(45, 1, strides=1, activation='relu', padding='valid')(hidden)
        hidden = Conv3D(45, 1, strides=1, activation='relu', padding='valid')(hidden) #DK ono more layer
        hidden = Conv3D(45, 1, strides=1, activation='relu', padding='valid')(hidden) #DK ono more layer
        watch4 = hidden
        hidden = hidden
        outputs = Conv3D(1, 1, strides=1, activation='relu', padding='valid', name='output')(hidden)
        if self._train:
            self._model = Model(inputs=inputs, outputs=[y, outputs])
        else:
            self._model = Model(inputs=inputs, outputs=[y, outputs])
            #self._model = Model(inputs=inputs, outputs=[watch1, watch2, watch3, y, outputs])


    def _conv3d_model(self, patch_size, convs=None):
        """
        Conv3D model.
        """
        if self._train:
            inputs = Input(shape=(3, 3, 3, self._nPWI)) #DK inputs = Input(shape=(patch_size, patch_size, patch_size, self._nPWI))
        else:
            (dim0, dim1, dim2) = (self._test_shape[0], self._test_shape[1], self._test_shape[2])
            inputs = Input(shape=(dim0, dim1, dim2, self._nPWI))
        hidden = Conv3D(45, 3, activation='relu', padding='valid')(inputs) #DK hidden = Conv3D(150, 3, activation='relu', padding='valid')(inputs)
        hidden = Conv3D(45, 1, activation='relu', padding='valid')(hidden)
        hidden = Conv3D(45, 1, activation='relu', padding='valid')(hidden)

        hidden = Conv3D(45, 1, activation='relu', padding='valid')(hidden)
        hidden = Conv3D(45, 1, activation='relu', padding='valid')(hidden)
        hidden = Conv3D(45, 1, activation='relu', padding='valid')(hidden)

        outputs = Conv3D(2, 1, activation='relu', padding='valid')(hidden) #DK number of out

        self._model = Model(inputs=inputs, outputs=outputs)


    def _sequence_model(self, patch_size, convs=False):
        """
        Sequence model.
        """
        inputs = Input(shape=(self._nPWI,))
        # Define hidden layer
        hidden = Dense(150, activation='relu')(inputs)
        for i in np.arange(self._layer  - 1):
            hidden = Dense(150, activation='relu')(hidden)

        hidden = Dropout(0.1)(hidden)

        # Define output layer
        if self._seg:
            outputs = Dense(9, name='output')(hidden)
        else:
            outputs = Dense(2, name='output')(hidden)   #DK number of out

        self._model = Model(inputs=inputs, outputs=outputs)

    def _staged_model(self, patch_size, convs=False):
        """
        Staged model
        """

        inputs = Input(shape=(self._nPWI,))

        # Define hidden layer
        hidden = Dense(45, activation='relu')(inputs)
        middle = Dense(45, activation='relu')(hidden)
        y = Dense(1, name='y')(middle)
        if self._con:
            hidden = Concatenate(axis=-1)([middle, y])

            hidden = Dense(45, activation='relu')(hidden)
        else:
            hidden = Dense(45, activation='relu')(middle)

        # Define output layer
        output = Dense(1, name='output')(hidden) #DK Dense(4, )

        self._model = Model(inputs=inputs, outputs=[y, output])

	#conv3d_staged, _conv3d_staged_model
    __model = {
        'sequence' : _sequence_model,
        'staged' : _staged_model,
        'conv3d' : _conv3d_model,
        'conv3d_staged' : _conv3d_staged_model
    }

    def model(self, optimizer, loss, patch_size, convs=False):
        """
        Generate model.
        """
        #print("Generating Model.................")
        self.__model[self._type](self, patch_size, convs=convs)
        self._model.summary()
        self._model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    def _sequence_train(self, data, label, nbatch, epochs, callbacks, shuffle, validation_data):


        validation_split = 0.0
        if validation_data is None:
            validation_split = 0.2

        print("DK check point in sequence_train -----------")
        self._hist = self._model.fit(data, label,
                                     batch_size=nbatch,
                                     epochs=epochs,
                                     shuffle=shuffle,
                                     validation_data=validation_data,
                                     validation_split=validation_split,
                                     callbacks=callbacks)


        self._loss.append(len(self._hist.history['loss']))
        self._loss.append(self._hist.history['loss'][-1])
        self._loss.append(None)
        self._loss.append(self._hist.history['acc'][-1])
        self._loss.append(None)

    def _staged_train(self, data, label, nbatch, epochs, callbacks, shuffle, validation_data):

        validation_split = 0.0
        if validation_data is not None:
            vdata, vlabel = validation_data
            validation_data = (vdata, [vlabel[..., :3], vlabel[..., 3:]])
        else:
            validation_split = 0.2

        self._hist = self._model.fit(data, [label[...,:1], label[..., 1:]],
                                     batch_size=nbatch,
                                     epochs=epochs,
									 #verbose=0,
                                     shuffle=shuffle,
                                     validation_data=validation_data,
                                     validation_split=validation_split,
                                     callbacks=callbacks)

        self._loss.append(len(self._hist.history['loss']))


        self._loss.append(self._hist.history['y_acc'][-1])
        self._loss.append(self._hist.history['output_acc'][-1])
        self._loss.append(self._hist.history['y_loss'][-1])
        self._loss.append(self._hist.history['output_loss'][-1])

	# conv3d_staged, _staged_train
    __train = {
        'sequence' : _sequence_train,
        'conv3d' : _sequence_train,
        'staged' : _staged_train,
        'conv3d_staged' : _staged_train
    }

    def train(self, data, label, nbatch, epochs, callbacks, weightname,
              shuffle=True, validation_data=None):

        self.__train[self._type](self, data, label, nbatch, epochs,
                                 callbacks, shuffle, validation_data)
        try:
            self._model.save_weights('weights/' + weightname + '.weights') #,save_format='h5'
        except IOError:
            os.system('mkdir weights')
            self._model.save_weights('weights/' + weightname + '.weights') #,save_format='h5'

        return self._loss

    def load_weight(self, weightname):

        self._model.load_weights('weights/' + weightname + '.weights')

    def predict(self, data):
        """
        Predict on test datas.
        """
        pred = self._model.predict(data)
        #if self._type[-6:] == 'staged':
        #    pred = np.concatenate((pred[0], pred[1]), axis=-1)

        return pred


def parser():
    """
    Create a parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--savename", metavar='name', help="Append the specific name")
    parser.add_argument("--PWI", metavar='N', help="Number of PWI", type=int, default=150)
    parser.add_argument("--base", metavar='base', help="choice of training data", type=int, default=1)
    parser.add_argument("--train", help="Train the network", action="store_true")
    parser.add_argument("--model", help="Train model",choices=['conv3d_staged'],default='conv3d_staged')
    parser.add_argument("--epoch", metavar='ep', help="Number of epoches", type=int, default=200)
    parser.add_argument("--train_subjects", help="Training sets orgs", nargs='*')
    parser.add_argument("--test_subject", help="Valid sets subs", nargs='*')
    parser.add_argument("--segment", help="Add segmentation mask to labels", action="store_true")
    parser.add_argument("--convs", help="Just for test", action="store_true")
    parser.add_argument("--loss", help="Just for test", action="store_true")
    parser.add_argument("--con", help="Just for test", action="store_true")


    return parser
