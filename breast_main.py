#-*-coding:utf-8-*-

from skimage import transform
from skimage import io

from self_fuse_resnet50_current import unet_model_3d
from keras.callbacks import ModelCheckpoint
from PIL import Image
from functools import partial

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
import scipy.io as scio

from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping

# Create Date：2019.7.22.
# Author：Dongdong Ma

bit_max_value = 4095
polar_img_num = 16          
resize_width = 128
resize_height = 128
aug_rotate_num = 12
aug_flip_num = 3  

aug_scale_num = 0   
aug_scale_step = 0.2
aug_translation_num = 0 
aug_translation_step = 5 

aug_blur_num = 0
aug_blur_step = 1   
blur_wsize = (5,5)

aug_gnoise_num = 0   
aug_gnoise_step = 0.002

mat_data_inner_name = 'cat_csv_data'

first_input_shape = (1,polar_img_num-1, resize_width, resize_height)
second_input_shape = (1,resize_width, resize_height)

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
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


def my_get_train_validate_test_label(data_path,class_names):
    data_lists = os.listdir(data_path)
    train_label = []
    validate_label = []
    test_label = []
    for data_list in data_lists:
        data_file_path = data_path + data_list
        sample_data = os.listdir(data_file_path)
        for mat_data in sample_data:

            cls_name = mat_data[0:7]
            if cls_name == class_names[0]:
                sample_label = 0
            elif cls_name == class_names[1]:
                sample_label = 1
            elif cls_name == class_names[2]:
                sample_label = 2
            else:
                sample_label = 3
            if data_list == 'train':
                train_label.append(sample_label)
            elif data_list == 'test':
                test_label.append(sample_label)
            else:
                validate_label.append(sample_label)
    return train_label, validate_label, test_label
    test_end = 1

def get_second_train_generator(train_data_path,train_batch_size):

    f = h5py.File(train_data_path,'r')
    train_data = f['data'][:]
    train_label = f['label'][:]
    train_first_image_data = f['first_image'][:]
    f.close()

    test1 = train_data[0, 0, 0,:,:]
    test2 = train_data[0, 0, 1, :, :]
    test3 = train_data[0, 0, 2, :, :]

    train_data_num = train_first_image_data.shape[0]
    num_train_batches = int(train_data_num/train_batch_size)

    while True:
        # permutation = list(np.random.permutation(validate_data_num))       
        permutation = list(range(train_data_num))                   
        for i in range(num_train_batches):
            permute_indexes = permutation[i*train_batch_size:(i+1)*train_batch_size]
            batch_train_data = train_data[permute_indexes, :, :, :, :]
            batch_train_first_image_data = train_first_image_data[permute_indexes,:,:,:]
            batch_train_label = train_label[permute_indexes,]
            yield [batch_train_label,batch_train_data, batch_train_first_image_data], \
                  [batch_train_label,batch_train_label]

def get_second_validate_generator(validate_data_path,validate_batch_size):

    f = h5py.File(validate_data_path,'r')
    validate_data = f['data'][:]
    validate_label = f['label'][:]
    validate_first_image_data = f['first_image'][:]
    f.close()

    validate_data_num = validate_first_image_data.shape[0]
    num_validate_batches = int(validate_data_num/validate_batch_size)

    while True:
        # permutation = list(np.random.permutation(validate_data_num))    
        permutation = list(range(validate_data_num))                  
        for i in range(num_validate_batches):
            permute_indexes = permutation[i*validate_batch_size:(i+1)*validate_batch_size]
            batch_validate_data = validate_data[permute_indexes, :, :, :, :]
            batch_validate_first_image_data = validate_first_image_data[permute_indexes,:,:,:]
            batch_validate_label = validate_label[permute_indexes,]
            yield [batch_validate_label,batch_validate_data, batch_validate_first_image_data], \
                  [batch_validate_label,batch_validate_label]  

def get_second_test_generator(test_data_path,test_batch_size):

    f = h5py.File(test_data_path,'r')
    test_data = f['data'][:]
    test_label = f['label'][:]
    test_first_image_data = f['first_image'][:]
    f.close()

    test_data_num = test_first_image_data.shape[0]
    num_test_batches = int(test_data_num/test_batch_size)

    while True:
        # permutation = list(np.random.permutation(test_data_num))        
        permutation = list(range(test_data_num))                        
        for i in range(num_test_batches):
            permute_indexes = permutation[i*test_batch_size:(i+1)*test_batch_size]
            batch_test_data = test_data[permute_indexes, :, :, :, :]
            batch_test_first_image_data = test_first_image_data[permute_indexes,:,:,:]
            batch_test_label = test_label[permute_indexes,]
            # print (batch_test_label)
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


def flip_rotate_aug_and_save_to_h5(base_path, mat_data_filename, class_names):
    aug_mat_path = base_path + 'aug_' + mat_data_filename + str(resize_width) + '_' + str(resize_height) + '_h5_rotate' + str(aug_rotate_num) + '_flip' + str(aug_flip_num) + '/'
    if not(os.path.exists(aug_mat_path)):
        os.makedirs(aug_mat_path)

    train_mat_path = base_path + mat_data_filename + '/train/'
    train_data_lists = os.listdir(train_mat_path)
    counter = 0
    for each_mat_name in train_data_lists:
        mat_path = train_mat_path + each_mat_name
        train_mat_data_info = sio.loadmat(mat_path)   

        cat_train_mat_data = np.zeros(shape=(resize_width, resize_height,polar_img_num),dtype='float64')
        for ii in range(polar_img_num):
            mat_data_inner_name = "FinalM" + str(ii//4+1) + str(ii%4+1)
            layer_train_mat_data = train_mat_data_info[mat_data_inner_name].astype('float64')
            layer_train_mat_data = layer_train_mat_data - layer_train_mat_data.min()
            layer_train_mat_data = layer_train_mat_data/layer_train_mat_data.max()  
            layer_train_mat_data = transform.resize(layer_train_mat_data, (resize_width, resize_height)) 
            cat_train_mat_data[:,:,ii] = layer_train_mat_data

        swap_resize_mat_data2 = cat_train_mat_data

        cls_name = each_mat_name[0:7]
        if cls_name == class_names[0]:
            train_mat_label = 0
        elif cls_name == class_names[1]:
            train_mat_label = 1
        elif cls_name == class_names[2]:
            train_mat_label = 2
        else:
            train_mat_label = 3
        train_mat_label = np.array([train_mat_label])

        for angle_num in range(aug_rotate_num):
            angle_step = 360/aug_rotate_num
            this_rotate_angle = angle_num*angle_step
            M = cv2.getRotationMatrix2D(((resize_width - 1) / 2, (resize_height - 1) / 2), this_rotate_angle, 1)
            rotated_mat_data = cv2.warpAffine(swap_resize_mat_data2, M, (resize_width, resize_height), borderMode=1)

            restore_mat_data1 = np.swapaxes(rotated_mat_data, 0, 2)
            restore_mat_data2 = np.swapaxes(restore_mat_data1, 1, 2)

            train_mat_data = np.expand_dims(restore_mat_data2, axis=0)
            train_mat_data = np.expand_dims(train_mat_data, axis=0)

            diff_train_mat_data = train_mat_data[:, :, 1:, :, :]
            first_resized_polar_image = train_mat_data[:, :, 0, :, :]
            cat_first_resized_polar_image = np.concatenate((first_resized_polar_image,first_resized_polar_image,first_resized_polar_image),axis = 1)
            cat_first_resized_polar_image = np.swapaxes(cat_first_resized_polar_image, 1, 3)
            cat_first_resized_polar_image = np.swapaxes(cat_first_resized_polar_image, 1, 2)

            if counter == 0:
                batch_diff_train_data = diff_train_mat_data
                batch_train_label = train_mat_label
                batch_first_resized_polar_image = cat_first_resized_polar_image
                counter = counter + 1

            else:
                batch_diff_train_data = np.concatenate((batch_diff_train_data, diff_train_mat_data), axis=0)
                batch_train_label = np.concatenate((batch_train_label, train_mat_label), axis=0)
                batch_first_resized_polar_image = np.concatenate((batch_first_resized_polar_image, cat_first_resized_polar_image), axis=0)

        fliped_swap_resize_mat_data2 = cv2.flip(swap_resize_mat_data2,0)
        restore_mat_data1 = np.swapaxes(fliped_swap_resize_mat_data2, 0, 2)
        restore_mat_data2 = np.swapaxes(restore_mat_data1, 1, 2)

        train_mat_data = np.expand_dims(restore_mat_data2, axis=0)
        train_mat_data = np.expand_dims(train_mat_data, axis=0)

        first_resized_polar_image = train_mat_data[:, :, 0, :, :]
        cat_first_resized_polar_image = np.concatenate((first_resized_polar_image, first_resized_polar_image, first_resized_polar_image), axis=1)
        cat_first_resized_polar_image = np.swapaxes(cat_first_resized_polar_image, 1, 3)
        cat_first_resized_polar_image = np.swapaxes(cat_first_resized_polar_image, 1, 2)

        diff_train_mat_data = train_mat_data[:, :, 1:, :, :]

        batch_diff_train_data = np.concatenate((batch_diff_train_data, diff_train_mat_data), axis=0)
        batch_train_label = np.concatenate((batch_train_label, train_mat_label), axis=0)
        batch_first_resized_polar_image = np.concatenate((batch_first_resized_polar_image, cat_first_resized_polar_image), axis=0)

        fliped_swap_resize_mat_data2 = cv2.flip(swap_resize_mat_data2,1)
        restore_mat_data1 = np.swapaxes(fliped_swap_resize_mat_data2, 0, 2)
        restore_mat_data2 = np.swapaxes(restore_mat_data1, 1, 2)

        train_mat_data = np.expand_dims(restore_mat_data2, axis=0)
        train_mat_data = np.expand_dims(train_mat_data, axis=0)

        first_resized_polar_image = train_mat_data[:, :, 0, :, :]
        cat_first_resized_polar_image = np.concatenate((first_resized_polar_image, first_resized_polar_image, first_resized_polar_image), axis=1)
        cat_first_resized_polar_image = np.swapaxes(cat_first_resized_polar_image, 1, 3)
        cat_first_resized_polar_image = np.swapaxes(cat_first_resized_polar_image, 1, 2)
        diff_train_mat_data = train_mat_data[:, :, 1:, :, :]

        batch_diff_train_data = np.concatenate((batch_diff_train_data, diff_train_mat_data), axis=0)
        batch_train_label = np.concatenate((batch_train_label, train_mat_label), axis=0)
        batch_first_resized_polar_image = np.concatenate((batch_first_resized_polar_image, cat_first_resized_polar_image), axis=0)

        fliped_swap_resize_mat_data2 = cv2.flip(swap_resize_mat_data2,-1)
        restore_mat_data1 = np.swapaxes(fliped_swap_resize_mat_data2, 0, 2)
        restore_mat_data2 = np.swapaxes(restore_mat_data1, 1, 2)

        train_mat_data = np.expand_dims(restore_mat_data2, axis=0) 
        train_mat_data = np.expand_dims(train_mat_data, axis=0)

        first_resized_polar_image = train_mat_data[:, :, 0, :, :]
        cat_first_resized_polar_image = np.concatenate((first_resized_polar_image, first_resized_polar_image, first_resized_polar_image), axis=1)
        cat_first_resized_polar_image = np.swapaxes(cat_first_resized_polar_image, 1, 3)
        cat_first_resized_polar_image = np.swapaxes(cat_first_resized_polar_image, 1, 2)
        diff_train_mat_data = train_mat_data[:, :, 1:, :, :]

        batch_diff_train_data = np.concatenate((batch_diff_train_data, diff_train_mat_data), axis=0)
        batch_train_label = np.concatenate((batch_train_label, train_mat_label), axis=0)
        batch_first_resized_polar_image = np.concatenate((batch_first_resized_polar_image, cat_first_resized_polar_image), axis=0)

    f = h5py.File(aug_mat_path + 'train_data_label.h5', 'w')
    f['data'] = batch_diff_train_data
    f['label'] = batch_train_label
    f['first_image'] = batch_first_resized_polar_image
    f.close()

    validate_mat_path = base_path + mat_data_filename + '/validate/'
    validate_data_lists = os.listdir(validate_mat_path)
    counter = 0
    for each_mat_name in validate_data_lists:
        mat_path = validate_mat_path + each_mat_name

        validate_mat_data_info = sio.loadmat(mat_path)
        cat_validate_mat_data = np.zeros(shape=(resize_width, resize_height,polar_img_num),dtype='float64')
        for ii in range(polar_img_num):
            mat_data_inner_name = "FinalM" + str(ii//4+1) + str(ii%4+1)
            layer_validate_mat_data = validate_mat_data_info[mat_data_inner_name].astype('float64')
            layer_validate_mat_data = layer_validate_mat_data - layer_validate_mat_data.min()
            layer_validate_mat_data = layer_validate_mat_data/layer_validate_mat_data.max()  
            layer_validate_mat_data = transform.resize(layer_validate_mat_data, (resize_width, resize_height))
            cat_validate_mat_data[:,:,ii] = layer_validate_mat_data
        swap_resize_mat_data2 = cat_validate_mat_data

        cls_name = each_mat_name[0:7]
        if cls_name == class_names[0]:
            validate_mat_label = 0
        elif cls_name == class_names[1]:
            validate_mat_label = 1
        elif cls_name == class_names[2]:
            validate_mat_label = 2
        else:
            validate_mat_label = 3
        validate_mat_label = np.array([validate_mat_label])

        for angle_num in range(aug_rotate_num):
            angle_step = 360/aug_rotate_num
            this_rotate_angle = angle_num*angle_step
            M = cv2.getRotationMatrix2D(((resize_width - 1) / 2, (resize_height - 1) / 2), this_rotate_angle, 1)
            rotated_mat_data = cv2.warpAffine(swap_resize_mat_data2, M, (resize_width, resize_height), borderMode=1)

            restore_mat_data1 = np.swapaxes(rotated_mat_data, 0, 2)
            restore_mat_data2 = np.swapaxes(restore_mat_data1, 1, 2)

            validate_mat_data = np.expand_dims(restore_mat_data2, axis=0)
            validate_mat_data = np.expand_dims(validate_mat_data, axis=0)

            diff_validate_mat_data = validate_mat_data[:, :, 1:, :, :]
            first_resized_polar_image = validate_mat_data[:, :, 0, :, :]
            cat_first_resized_polar_image = np.concatenate(
                (first_resized_polar_image, first_resized_polar_image, first_resized_polar_image), axis=1)
            cat_first_resized_polar_image = np.swapaxes(cat_first_resized_polar_image, 1, 3)
            cat_first_resized_polar_image = np.swapaxes(cat_first_resized_polar_image, 1, 2)

            if counter == 0:
                batch_diff_validate_data = diff_validate_mat_data
                batch_validate_label = validate_mat_label
                batch_first_resized_polar_image = cat_first_resized_polar_image
                counter = counter + 1
            else:
                batch_diff_validate_data = np.concatenate((batch_diff_validate_data, diff_validate_mat_data), axis=0)
                batch_validate_label = np.concatenate((batch_validate_label, validate_mat_label), axis=0)
                batch_first_resized_polar_image = np.concatenate((batch_first_resized_polar_image, cat_first_resized_polar_image), axis=0)

        fliped_swap_resize_mat_data2 = cv2.flip(swap_resize_mat_data2,0)
        restore_mat_data1 = np.swapaxes(fliped_swap_resize_mat_data2, 0, 2)
        restore_mat_data2 = np.swapaxes(restore_mat_data1, 1, 2)

        validate_mat_data = np.expand_dims(restore_mat_data2, axis=0)
        validate_mat_data = np.expand_dims(validate_mat_data, axis=0)

        first_resized_polar_image = validate_mat_data[:, :, 0, :, :]
        cat_first_resized_polar_image = np.concatenate(
            (first_resized_polar_image, first_resized_polar_image, first_resized_polar_image), axis=1)
        cat_first_resized_polar_image = np.swapaxes(cat_first_resized_polar_image, 1, 3)
        cat_first_resized_polar_image = np.swapaxes(cat_first_resized_polar_image, 1, 2)
        diff_validate_mat_data = validate_mat_data[:, :, 1:, :, :]

        batch_diff_validate_data = np.concatenate((batch_diff_validate_data, diff_validate_mat_data), axis=0)
        batch_validate_label = np.concatenate((batch_validate_label, validate_mat_label), axis=0)
        batch_first_resized_polar_image = np.concatenate((batch_first_resized_polar_image, cat_first_resized_polar_image), axis=0)

        fliped_swap_resize_mat_data2 = cv2.flip(swap_resize_mat_data2,1)
        restore_mat_data1 = np.swapaxes(fliped_swap_resize_mat_data2, 0, 2)
        restore_mat_data2 = np.swapaxes(restore_mat_data1, 1, 2)

        validate_mat_data = np.expand_dims(restore_mat_data2, axis=0)
        validate_mat_data = np.expand_dims(validate_mat_data, axis=0)

        first_resized_polar_image = validate_mat_data[:, :, 0, :, :]
        cat_first_resized_polar_image = np.concatenate(
            (first_resized_polar_image, first_resized_polar_image, first_resized_polar_image), axis=1)
        cat_first_resized_polar_image = np.swapaxes(cat_first_resized_polar_image, 1, 3)
        cat_first_resized_polar_image = np.swapaxes(cat_first_resized_polar_image, 1, 2)
        diff_validate_mat_data = validate_mat_data[:, :, 1:, :, :]

        batch_diff_validate_data = np.concatenate((batch_diff_validate_data, diff_validate_mat_data), axis=0)
        batch_validate_label = np.concatenate((batch_validate_label, validate_mat_label), axis=0)
        batch_first_resized_polar_image = np.concatenate(
            (batch_first_resized_polar_image, cat_first_resized_polar_image), axis=0)

        fliped_swap_resize_mat_data2 = cv2.flip(swap_resize_mat_data2,-1)
        restore_mat_data1 = np.swapaxes(fliped_swap_resize_mat_data2, 0, 2)
        restore_mat_data2 = np.swapaxes(restore_mat_data1, 1, 2)

        validate_mat_data = np.expand_dims(restore_mat_data2, axis=0)
        validate_mat_data = np.expand_dims(validate_mat_data, axis=0)

        first_resized_polar_image = validate_mat_data[:, :, 0, :, :]
        cat_first_resized_polar_image = np.concatenate(
            (first_resized_polar_image, first_resized_polar_image, first_resized_polar_image), axis=1)
        cat_first_resized_polar_image = np.swapaxes(cat_first_resized_polar_image, 1, 3)
        cat_first_resized_polar_image = np.swapaxes(cat_first_resized_polar_image, 1, 2)
        diff_validate_mat_data = validate_mat_data[:, :, 1:, :, :]

        batch_diff_validate_data = np.concatenate((batch_diff_validate_data, diff_validate_mat_data), axis=0)
        batch_validate_label = np.concatenate((batch_validate_label, validate_mat_label), axis=0)
        batch_first_resized_polar_image = np.concatenate((batch_first_resized_polar_image, cat_first_resized_polar_image), axis=0)

    f = h5py.File(aug_mat_path + 'validate_data_label.h5', 'w')
    f['data'] = batch_diff_validate_data
    f['label'] = batch_validate_label
    f['first_image'] = batch_first_resized_polar_image
    f.close()

    test_mat_path = base_path + mat_data_filename + '/test/'
    test_data_lists = os.listdir(test_mat_path)
    counter = 0
    for each_mat_name in test_data_lists:
        mat_path = test_mat_path + each_mat_name

        test_mat_data_info = sio.loadmat(mat_path)
        cat_test_mat_data = np.zeros(shape=(resize_width, resize_height,polar_img_num),dtype='float64')
        for ii in range(polar_img_num):
            mat_data_inner_name = "FinalM" + str(ii//4+1) + str(ii%4+1)
            layer_test_mat_data = test_mat_data_info[mat_data_inner_name].astype('float64')
            layer_test_mat_data = layer_test_mat_data - layer_test_mat_data.min()
            layer_test_mat_data = layer_test_mat_data/layer_test_mat_data.max()  
            layer_test_mat_data = transform.resize(layer_test_mat_data, (resize_width, resize_height)) 
            cat_test_mat_data[:,:,ii] = layer_test_mat_data

        resize_test_mat_data = cat_test_mat_data
        restore_mat_data1 = np.swapaxes(resize_test_mat_data, 0, 2)
        restore_mat_data2 = np.swapaxes(restore_mat_data1, 1, 2)

        test_mat_data = np.expand_dims(restore_mat_data2, axis=0)
        test_mat_data = np.expand_dims(test_mat_data, axis=0)
        diff_test_mat_data = test_mat_data[:, :, 1:, :, :]

        cls_name = each_mat_name[0:7]
        if cls_name == class_names[0]:
            test_mat_label = 0
        elif cls_name == class_names[1]:
            test_mat_label = 1
        elif cls_name == class_names[2]:
            test_mat_label = 2
        else:
            test_mat_label = 3
        test_mat_label = np.array([test_mat_label])

        first_resized_polar_image = test_mat_data[:, :, 0, :, :]
        cat_first_resized_polar_image = np.concatenate(
            (first_resized_polar_image, first_resized_polar_image, first_resized_polar_image), axis=1)
        cat_first_resized_polar_image = np.swapaxes(cat_first_resized_polar_image, 1, 3)
        cat_first_resized_polar_image = np.swapaxes(cat_first_resized_polar_image, 1, 2)

        if counter == 0:
            batch_diff_test_data = diff_test_mat_data
            batch_test_label = test_mat_label
            batch_first_resized_polar_image = cat_first_resized_polar_image
        else:
            batch_diff_test_data = np.concatenate((batch_diff_test_data, diff_test_mat_data), axis=0)
            batch_test_label = np.concatenate((batch_test_label, test_mat_label), axis=0)
            batch_first_resized_polar_image = np.concatenate((batch_first_resized_polar_image, cat_first_resized_polar_image), axis=0)
        counter = counter + 1

    f = h5py.File(aug_mat_path + 'test_data_label.h5', 'w')
    f['data'] = batch_diff_test_data
    f['label'] = batch_test_label
    f['first_image'] = batch_first_resized_polar_image
    f.close()

    test_end = 1

def step_decay(epoch, initial_lrate, drop, epochs_drop):
    return initial_lrate * math.pow(drop, math.floor((1+epoch)/float(epochs_drop)))

# 10.03, the best weight is 0.00007,0.5,30
def get_callbacks(initial_learning_rate = 0.0003, learning_rate_drop = 0.5, learning_rate_epochs = 30,
                  learning_rate_patience = 2, logging_file = "training.log", verbosity = 1,
                  early_stopping_patience = None):

    callbacks = list()

    model_checkpoint = ModelCheckpoint('weights/' + 'weight.{epoch:02d}-{val_softmax_acc:.4f}.hdf5',
                                       monitor = 'val_softmax_acc', verbose=0, save_best_only=False, mode='auto')
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

    train_num = 109*(aug_rotate_num + aug_flip_num + aug_scale_num + aug_translation_num + aug_blur_num + aug_gnoise_num)
    validate_num = 54*(aug_rotate_num + aug_flip_num + aug_scale_num + aug_translation_num + aug_blur_num + aug_gnoise_num)
    test_num = 108
    class_names = ["0000231", "00bt474", "000mcf7", "sk-br-3"]
    base_path = '/media/DATA3/mdd/modified_polar_cell_data3/'

    mat_data_filename = 'random_select1_MM/'
    mat_data_path = base_path + mat_data_filename

    # flip & rotate
    aug_h5_data_path = base_path + 'aug_' + mat_data_filename + str(resize_width) + '_' + str(resize_height) + '_h5_rotate' + str(aug_rotate_num) + '_flip' + str(aug_flip_num) + '/'
    if not os.path.isdir(aug_h5_data_path):
        os.makedirs(aug_h5_data_path)
    # flip_rotate_aug_and_save_to_h5(base_path, mat_data_filename, class_names)

    concat_model,concat_test_model = unet_model_3d(first_input_shape, 4)
    concat_model.summary()

    train_label, validate_label, test_label = my_get_train_validate_test_label(mat_data_path, class_names)
    train_batch_size = 54
    validate_batch_size = 27
    test_batch_size = 54

    second_train_generator, second_validate_generator, second_test_generator = get_second_training_validate_test_generators(aug_h5_data_path,train_batch_size,validate_batch_size,test_batch_size)

    train_or_test = 1      
    all_result_accuracy = []
    if train_or_test == 1:
       concat_model.compile(optimizer='adam', loss=['sparse_categorical_crossentropy', lambda y_true, y_pred: y_pred],
                            loss_weights=[1., 0.001], metrics={"softmax":'accuracy'}) 
       concat_model.fit_generator(generator = second_train_generator, steps_per_epoch = int(train_num/train_batch_size), epochs = 500,
                                  validation_data = second_validate_generator,
                                  validation_steps = int(validate_num/validate_batch_size),
                                  callbacks = get_callbacks())
    elif train_or_test == 0:
       weight_filename = '/home/mdd/work/3DUNet/recode/self_3DNet_v7_success/best_weight/'
       weight_list = os.listdir(weight_filename)
       for weight_name in weight_list:
           print(weight_name)
           concat_test_model.compile(optimizer='adam', loss=['sparse_categorical_crossentropy'],
                                     metrics=['accuracy'])
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






