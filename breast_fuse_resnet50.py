#-*-coding:utf-8-*-
import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, Conv2D, MaxPooling2D, Flatten, Dense, UpSampling3D, Activation, BatchNormalization, PReLU, Deconvolution3D, Lambda, Embedding
from keras.layers import Dropout,add,Reshape, GlobalAveragePooling2D,Multiply,Lambda,Add,Average,GlobalAveragePooling3D,GlobalMaxPooling3D,Permute,Subtract
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

# K.set_image_data_format("channels_first")

try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate

# loss函数1：center loss
def center_loss(x):
   # numerator = K.sum(K.square(x[0] - x[1][:, 0, :]), 1, keepdims = True)
   numerator = K.sum(K.square(x[0] - x[1][:, 0, :]), 1)

   return numerator

def fuse(input):
    data = input[0]
    weight = input[1]
    weight_expand1 = K.expand_dims(weight, -2)
    weight_expand2 = K.expand_dims(weight_expand1, -2)
    weight_expand3 = K.expand_dims(weight_expand2, -2)

    weight_repeat1 = K.repeat_elements(weight_expand3, data.shape[1],  axis=1)
    weight_repeat2 = K.repeat_elements(weight_repeat1, data.shape[2], axis=2)
    weight_repeat3 = K.repeat_elements(weight_repeat2, data.shape[3], axis=3)

    fused_multiply = data*weight_repeat3
    fused_sum = K.sum(fused_multiply,4)
    fused_sum = K.sum(fused_sum, 1)
    fused_sum = K.expand_dims(fused_sum, -1)

    return fused_sum

def feature_weighting(input):
    data = input[0]
    weight = input[1]
    weight_expand1 = K.expand_dims(weight,-1)
    weight_expand2 = K.expand_dims(weight_expand1,-1)
    weight_repeat1 = K.repeat_elements(weight_expand2,data.shape[1],axis=1)
    weight_repeat2 = K.repeat_elements(weight_repeat1, data.shape[2], axis=2)

    weighted_feat = data*weight_repeat2
    # weighted_feat = K.squeeze(weighted_feat,axis=3)

    return weighted_feat

# loss函数2：consistence loss
def rcosine_loss(input):
    dense_feat = input[0]
    backbone_dense_feat = input[1]

    square_dense_feat = dense_feat*dense_feat
    square_backbone_dense_feat = backbone_dense_feat * backbone_dense_feat

    sumsquare_dense_feat = K.sum(square_dense_feat, axis=1)
    sumsquare_backbone_dense_feat = K.sum(square_backbone_dense_feat, axis=1)

    norminator = dense_feat*backbone_dense_feat
    norminator = K.sum(norminator,axis=1)

    denominator = sumsquare_dense_feat*sumsquare_backbone_dense_feat
    denominator = denominator**0.5

    similarity = norminator/denominator
    return  1 - similarity


def unet_model_3d(first_input_shape, nb_classes):

    channel_first_polar_input = Input(first_input_shape)
    first_input = Permute([2, 3, 4, 1])(channel_first_polar_input)

    conv_layer2 = Conv3D(8, (5, 5, 5), padding='same', activation='relu')(first_input)
    conv_layer3 = Conv3D(8, (3, 3, 3), padding='same', activation='relu')(conv_layer2)

    # attention branch1
    attention1_permute = Permute([4, 2, 3, 1])(first_input)
    attention1_gpooling = GlobalAveragePooling3D()(attention1_permute)
    attention1_dense = Dense(units = 15, activation='relu')(attention1_gpooling)
    branch1 = Lambda(fuse)([attention1_permute, attention1_dense])

    # attention branch2
    attention2_permute = Permute([4, 2, 3, 1])(conv_layer2)
    attention2_gpooling = GlobalAveragePooling3D()(attention2_permute)
    attention2_dense = Dense(units = 15, activation='relu')(attention2_gpooling)
    branch2 = Lambda(fuse)([attention2_permute, attention2_dense])

    # attention branch3
    attention3_permute = Permute([4, 2, 3, 1])(conv_layer3)
    attention3_gpooling = GlobalAveragePooling3D()(attention3_permute)
    attention3_dense = Dense(units = 15, activation='relu')(attention3_gpooling)
    branch3 = Lambda(fuse)([attention3_permute, attention3_dense])

    # branch1_weighting
    branch_gpooling1 = MaxPooling2D(strides=4)(branch1)
    branch_flatten1 = Flatten()(branch_gpooling1)
    branch_dense1 = Dense(units = 1, activation='relu')(branch_flatten1)
    weighted_branch1 = Lambda(feature_weighting)([branch1, branch_dense1])

    # branch2_weighting
    branch_gpooling2 = MaxPooling2D(strides=4)(branch2)
    branch_flatten2 = Flatten()(branch_gpooling2)
    branch_dense2 = Dense(units = 1, activation='relu')(branch_flatten2)
    weighted_branch2 = Lambda(feature_weighting)([branch2, branch_dense2])

    # branch3_weighting
    branch_gpooling3 = MaxPooling2D(strides=4)(branch3)
    branch_flatten3 = Flatten()(branch_gpooling3)
    branch_dense3 = Dense(units = 1, activation='relu')(branch_flatten3)
    weighted_branch3 = Lambda(feature_weighting)([branch3, branch_dense3])

    # fuse_output
    fused_feat = Add()([weighted_branch1, weighted_branch2, weighted_branch3])

    conv_fused_feat = Conv2D(8, (3, 3), padding='same', activation='relu',strides=4)(fused_feat)
    # res_block1
    res_block1_conv1 = Conv2D(8, (3, 3), padding='same', activation='relu')(conv_fused_feat)
    res_block1_bnorm1 = BatchNormalization()(res_block1_conv1)
    res_block1_conv2 = Conv2D(8, (3, 3), padding='same', activation='relu')(res_block1_bnorm1)
    res_block1_output = Add()([res_block1_conv2,conv_fused_feat])
    res_block1_bnorm2 = BatchNormalization()(res_block1_output)
    res_block1_pooling = MaxPooling2D()(res_block1_bnorm2)
    # res_block2
    res_block2_conv1 = Conv2D(8, (3, 3), padding='same', activation='relu')(res_block1_pooling)
    res_block2_bnorm1 = BatchNormalization()(res_block2_conv1)
    res_block2_conv2 = Conv2D(8, (3, 3), padding='same', activation='relu')(res_block2_bnorm1)
    res_block2_output = Add()([res_block2_conv2,res_block1_pooling])
    res_block2_bnorm2 = BatchNormalization()(res_block2_output)
    res_block2_pooling = MaxPooling2D()(res_block2_bnorm2)
    # res_block3
    res_block3_conv1 = Conv2D(8, (3, 3), padding='same', activation='relu')(res_block2_pooling)
    res_block3_bnorm1 = BatchNormalization()(res_block3_conv1)
    res_block3_conv2 = Conv2D(8, (3, 3), padding='same', activation='relu')(res_block3_bnorm1)
    res_block3_output = Add()([res_block3_conv2,res_block2_pooling])
    res_block3_bnorm2 = BatchNormalization()(res_block3_output)
    res_block3_pooling = MaxPooling2D()(res_block3_bnorm2)

    # flatten
    flatten_feat = Flatten()(res_block3_pooling)
    dense_feat = Dense(units = 256, activation='linear')(flatten_feat)

    base_model = ResNet50(weights='imagenet',include_top=False,input_shape=[128,128,3])
    second_input = base_model.input
    resnet50_activation_98_output = base_model.output
    resnet50_gpooling = GlobalAveragePooling2D()(resnet50_activation_98_output)
    backbone_dense_feat = Dense(256, activation='relu')(resnet50_gpooling)


    # concat_feat
    concat_layer = concatenate([dense_feat, backbone_dense_feat],axis = 1)

    # polar loss
    input_target = Input(shape=(1,))
    centers = Embedding(nb_classes, 512)(input_target)
    l2_loss = Lambda(center_loss, name='l2_loss')([concat_layer, centers])
    # similarity loss
    nonsimilarity_loss = Lambda(rcosine_loss, name='consistence_loss')([dense_feat, backbone_dense_feat])
    # consistence loss
    consistence_loss = Multiply()([nonsimilarity_loss, l2_loss])

    # output1
    softmax_output1 = Dense(units = nb_classes, activation = 'softmax',name = 'softmax1')(dense_feat)
    # output2
    softmax_output2 = Dense(units = nb_classes, activation = 'softmax',name = 'softmax2')(backbone_dense_feat)
    # average softmax_output1 and softmax_output2

    softmax_output = Average(name = 'softmax')([softmax_output1, softmax_output2])

    train_model = Model(inputs=[input_target,channel_first_polar_input,second_input],
                        outputs=[softmax_output, consistence_loss])
    test_model = Model(inputs=[channel_first_polar_input,second_input],
                        outputs=[softmax_output])

    # return model_train, model_test
    return train_model, test_model


