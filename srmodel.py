import logging
import tensorflow as tf
import keras.backend as K
from keras.callbacks import Callback, LearningRateScheduler
from keras.layers import Conv2D, Input, Lambda, Add
from keras.models import Model
from keras.initializers import glorot_normal
from keras.optimizers import Adam
from keras.losses import MeanAbsoluteError
from keras.models import load_model, clone_model
import tensorflow_model_optimization as tfmot
import numpy as np
import math
import pickle
import os
import sys
import time
import shutil
import cv2
import random
from tensorboardX import SummaryWriter

# Notes: 
## Will need to use tensorflow keras functional api as the ABRL layer appears to have two inputs
## Won't use command line options, hardcode values and auto-check for resumption of training

# Important Values
# ================

# Network settings
conv_kernel_size = 3
num_deep_feature_layers = 5
num_kernels = 28

# Training settings general
training_batch_size = 16
lr_image_size = (64, 64)

# Initial training settings
learning_rate = 0.001
lr_reduction_interval = 200
num_epochs = 1000

# QAT settings
qat_learning_rate = 0.0001
qat_lr_reduction_interval = 50
qat_num_epochs = 200

# Data
training_data = [] # TODO Get the data here

# Scale is assumed to be hardcoded at 3 for now, this could be changed though

# Network Architecture
# ====================

def ABmodel():
    # Input
    inputs = Input(shape=(None, None, 3))
    upsampled = tf.concat([inputs]*(3**2), axis=3)

    # Shallow Feature Extraction, single 3x3 Convolution layer followed by ReLU
    x = Conv2D(num_kernels, conv_kernel_size, activation='relu', padding='same')(inputs)

    # Deep Feature Extraction, 5 layers of 3x3 Convolution layers paired with ReLUs
    for _ in range(num_deep_feature_layers):
        x = Conv2D(num_kernels, conv_kernel_size, activation='relu', padding='same')(x)

    # Transfer features to HR image space, one convolution layer (Assume from the paper's figure it is 3x3 conv)
    x = Conv2D(3*(3**2), conv_kernel_size, padding='same')(x) # This layer needs to have an output with the same shape as upsampled

    # Apply ABRL
    x = Add()([upsampled, x])

    # Reshuffle Pixels, labelled as depth to space in paper's figure
    x = tf.nn.depth_to_space(x, 3)

    # Clip node to restrict pixel values to the range 0 to 255, this gives final output
    outputs = K.clip(x, 0., 255.)

    return Model(inputs=inputs, outputs=outputs, name='anchor_based_plain_net')

# Model Training
# ==============

def scheduler(epoch, lr):
    if epoch % lr_reduction_interval == 0:
        lr = lr*0.5
    return lr

def train():
    model = ABmodel() 
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=MeanAbsoluteError())
    model.fit(training_data, callbacks=[LearningRateScheduler(scheduler)], epochs=num_epochs)

class NoOpQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    def get_weights_and_quantizers(self, layer):
        return []
    def get_activations_and_quantizers(self, layer):
        return []
    def set_quantize_weights(self, layer, quantize_weights):
        pass
    def set_quantize_activations(self, layer, quantize_anctivations):
        pass
    def get_output_quantizers(self, layer):
        return []
    def get_config(self):
        return {}

def qat_helper(layer):
    if 'concat' in layer.name or 'depth' in layer.name or 'clip' in layer.name:
        return tfmot.quantization.keras.quantize_annotate_layer(layer, quantize_config=NoOpQuantizeConfig())
    return layer

def qat_scheduler(epoch, lr):
    if epoch % qat_lr_reduction_interval == 0:
        lr = lr*0.5
    return lr

def qat_train(model_path):
    model = load_model(model_path)
    model = clone_model(model, clone_function=qat_helper)
    model = tfmot.quantization.keras.quantize_annotate_model(model)
    dts = tf.nn.depth_to_space(x, 3)
    with tfmot.quantization.keras.quantize_scope({'NoOpQuantizeConfig': NoOpQuantizeConfig, 'depth_to_space': dts, 'tf': tf}):
        model = tfmot.quantization.keras.quantize_apply(model)
    model.compile(optimizer=Adam(learning_rate=qat_learning_rate), loss=MeanAbsoluteError())
    model.fit(training_data, callbacks=[LearningRateScheduler(scheduler)], epochs=num_epochs)

if __name__ == "__main__":
    model = ABmodel()
    model.summary()