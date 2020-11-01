#-*-coding:utf-8-*-
from skimage import transform
from skimage import io

from fuse_resnet50 import unet_model_3d
from keras.callbacks import ModelCheckpoint
from PIL import Image
from functools import partial
import scipy.io as scio
from sklearn.manifold import TSNE

import os
import keras
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib
import skimage
import h5py
import cv2, shutil
import math

from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping

# Create Date：2019.7.22
# Author：Dongdong Ma

polar_img_num = 16
resize_width = 128
resize_height = 128
aug_rotate_num = 12
aug_flip_num = 0             

first_input_shape = (1,polar_img_num-1, resize_width, resize_height)
second_input_shape = (1,resize_width, resize_height)

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
config = dict()
config["pool_size"] = (2, 2)         
config["image_shape"] = (polar_img_num-1, resize_width, resize_height)  
config["all_modalities"] = ["polar"]
config["training_modalities"] = config["all_modalities"]    
config["nb_channels"] = len(config["training_modalities"])
config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))
config["truth_channel"] = config["nb_channels"]
config["deconvolution"] = True                    
config["n_epochs"] = 500                           
config["patience"] = 10                               
config["early_stop"] = 50                           
config["initial_learning_rate"] = 0.1
config["learning_rate_drop"] = 0.5                  
config["validation_split"] = 0.8                      
config["overwrite"] = False                           
num_classes = 4


def my_get_test_label(test_data_label_path):
    test_label_path = test_data_label_path + 'test_data_label.h5'
    f = h5py.File(test_label_path,'r')
    test_label = f['label'][:]
    f.close()
    return test_label

def get_second_train_generator(train_data_path,train_batch_size):    

    f = h5py.File(train_data_path,'r')
    train_data = f['data'][:]
    train_label = f['label'][:]
    f.close()

    train_data_num = train_data.shape[0]
    num_train_batches = int(train_data_num/train_batch_size)

    while True:
        permutation = list(np.random.permutation(train_data_num))            
        # permutation = list(range(train_data_num))                      
        for i in range(num_train_batches):
            permute_indexes = permutation[i*train_batch_size:(i+1)*train_batch_size]
            batch_train_data = train_data[permute_indexes, :, 1:, :, :]
            batch_train_first_image_data = np.repeat(train_data[permute_indexes, :, 0, :, :], 3, axis = 1)
            batch_train_first_image_data = batch_train_first_image_data.swapaxes(1, 3)
            batch_train_first_image_data = batch_train_first_image_data.swapaxes(1, 2)
            batch_train_label = train_label[permute_indexes,]
            yield [batch_train_label, batch_train_data, batch_train_first_image_data], [batch_train_label,batch_train_label]


def get_second_validate_generator(validate_data_path,validate_batch_size):

    f = h5py.File(validate_data_path,'r')
    validate_data = f['data'][:]
    validate_label = f['label'][:]
    f.close()

    validate_data_num = validate_data.shape[0]
    num_validate_batches = int(validate_data_num/validate_batch_size)

    while True:
        permutation = list(np.random.permutation(validate_data_num))
        # permutation = list(range(validate_data_num))        
        for i in range(num_validate_batches):
            permute_indexes = permutation[i*validate_batch_size:(i+1)*validate_batch_size]
            batch_validate_data = validate_data[permute_indexes, :, 1:, :, :]
            batch_validate_first_image_data = np.repeat(validate_data[permute_indexes, :, 0, :, :], 3, axis=1)
            batch_validate_first_image_data = batch_validate_first_image_data.swapaxes(1, 3)
            batch_validate_first_image_data = batch_validate_first_image_data.swapaxes(1, 2)
            batch_validate_label = validate_label[permute_indexes,]
            yield [batch_validate_label,batch_validate_data, batch_validate_first_image_data], \
                  [batch_validate_label,batch_validate_label]


def get_second_test_generator(test_data_path,test_batch_size):

    f = h5py.File(test_data_path,'r')
    test_data = f['data'][:]
    test_label = f['label'][:]
    f.close()

    test_data_num = test_data.shape[0]
    num_test_batches = int(test_data_num/test_batch_size)

    while True:
        # permutation = list(np.random.permutation(test_data_num))
        permutation = list(range(test_data_num)) 
        for i in range(num_test_batches):
            permute_indexes = permutation[i*test_batch_size:(i+1)*test_batch_size]
            batch_test_data = test_data[permute_indexes, :, 1:, :, :]
            batch_test_first_image_data = np.repeat(test_data[permute_indexes, :, 0, :, :], 3, axis=1)
            batch_test_first_image_data = batch_test_first_image_data.swapaxes(1, 3)
            batch_test_first_image_data = batch_test_first_image_data.swapaxes(1, 2)
            batch_test_label = test_label[permute_indexes,]
            yield [batch_test_data, batch_test_first_image_data], \
                  [batch_test_label] 


def get_second_training_validate_test_generators(data_path,train_batch_size,validate_batch_size,test_batch_size):

    train_data_file = data_path + 'train_data_label.h5'
    validate_data_file = data_path + 'validate_data_label.h5'
    test_data_file = data_path + 'test_data_label.h5'

    train_generator = get_second_train_generator(train_data_file,train_batch_size)
    validate_generator = get_second_validate_generator(validate_data_file,validate_batch_size)
    test_generator = get_second_test_generator(test_data_file,test_batch_size)

    return train_generator, validate_generator, test_generator

def step_decay(epoch, initial_lrate, drop, epochs_drop):
    return initial_lrate * math.pow(drop, math.floor((1+epoch)/float(epochs_drop)))

# 10.03, the best weight is 0.00007,0.5,30
def get_callbacks(initial_learning_rate = 0.0003, learning_rate_drop = 0.5, learning_rate_epochs = 30,
                  learning_rate_patience = 2, logging_file = "training.log", verbosity = 1,
                  early_stopping_patience = None):

    callbacks = list()

    model_checkpoint = ModelCheckpoint('weights/' + 'weight.{epoch:03d}-{loss:.4f}-{val_loss:.4f}.hdf5',
                                       monitor='val_loss', verbose = 0, save_best_only = False, mode = 'auto')
    callbacks.append(model_checkpoint)
    callbacks.append(CSVLogger(logging_file, append=True))
    if learning_rate_epochs:
        callbacks.append(LearningRateScheduler(partial(step_decay, initial_lrate=initial_learning_rate,
                                                       drop=learning_rate_drop, epochs_drop=learning_rate_epochs)))
    else:
        callbacks.append(ReduceLROnPlateau(factor=learning_rate_drop, patience=learning_rate_patience,
                                           verbose=verbosity)) 
    if early_stopping_patience:
        callbacks.append(EarlyStopping(verbose=verbosity, patience=early_stopping_patience))
    return callbacks

def self_main(overwrite = False):

    train_num = 597
    validate_num = 300
    test_num = 597
    class_names = ["G1000", "G1001", "G1002"]
    base_path = '/media/DATA3/mdd/modified_polar_cell_data3/'

    # flip & rotate
    aug_h5_data_path = '/media/DATA3/mdd/海藻数据/整理后/h5file-3类/train_validate_test_MM/'
    # flip_rotate_aug_and_save_to_h5(base_path, mat_data_filename, class_names)

    concat_model,concat_test_model = unet_model_3d(first_input_shape, 3)
    # concat_model.summary()
    test_label = my_get_test_label(aug_h5_data_path)

    train_batch_size = 50
    validate_batch_size = 30
    test_batch_size = 3
    second_train_generator, second_validate_generator, second_test_generator = get_second_training_validate_test_generators(aug_h5_data_path,train_batch_size,validate_batch_size,test_batch_size)

    train_or_test = 0
    all_result_accuracy = []
    if train_or_test == 1:
       concat_model.compile(optimizer = 'adam', loss=['sparse_categorical_crossentropy', lambda y_true, y_pred: y_pred],
                            loss_weights=[1., 0.0002], metrics={'softmax':"accuracy"})     #
       concat_model.fit_generator(generator = second_train_generator, steps_per_epoch = int(train_num/train_batch_size), epochs = 101,
                                  validation_data = second_validate_generator,
                                  validation_steps = int(validate_num/validate_batch_size),
                                  callbacks = get_callbacks())
    elif train_or_test == 0:
       concat_test_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
       weight_filename = 'resnet_best_weight/'
       weight_list = os.listdir(weight_filename)
       for weight_name in weight_list:
           print(weight_name)
           concat_model.load_weights(weight_filename + weight_name)

           pred_label_or_feature = 0    
           if pred_label_or_feature == 0:
              results = concat_test_model.evaluate_generator(second_test_generator, int(test_num / test_batch_size),verbose=0)

              print(results[1])
              all_result_accuracy.append(results[1])

    print(np.max(all_result_accuracy))
    test_end = 1

if __name__ == "__main__":

    np.random.seed(1337)
    self_main(overwrite=config["overwrite"])
