import argparse
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

# Helper Functions
# ================

# .pt files ostensibly load faster than .png files
def png2pt(path, pt_path, namelist):
	pngs = os.listdir(path)
	if os.path.exists(pt_path):
		# Check if the directory has all pt files with proper names
		pt_files = os.listdir(pt_path)
		for name in namelist:
			if name not in pt_files:
				break # Missing file, need to convert
			return # All files are present, no need to convert
	# Create the directory if it doesn't exist
	if not os.path.exists(pt_path):
		os.makedirs(pt_path)
	# Convert the images to .pt
	for png in pngs:
		base, ext = os.path.splitext(png)
		if ext == '.png':
			src = os.path.join(path, png)
			dst = os.path.join(pt_path, base + '.pt')
			with open(dst, 'wb') as f:
				img = cv2.imread(src)
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				pickle.dump(img, f)

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

# Data pre-processing
hrpath_t = 'data/DIV2K_train_HR'
hrpath_v = 'data/DIV2K_valid_HR'
lrpath_t = 'data/DIV2K_train_LR_bicubic/X3'
lrpath_v = 'data/DIV2K_valid_LR_bicubic/X3'
pt_pathhr = 'data/DIV2K_HR_pt'
pt_pathlr = 'data/DIV2K_LR_pt'
trainlisthr = ['%04d.pt' % i for i in range(1, 801)] # '0001.pt' to '0800.pt'
validlisthr = ['%04d.pt' % i for i in range(801, 901)] # '0801.pt' to '0900.pt'
trainlistlr = ['%04dx3.pt' % i for i in range(1, 801)] # '0001x3.pt' to '0800x3.pt'
validlistlr = ['%04dx3.pt' % i for i in range(801, 901)] # '0801x3.pt' to '0900x3.pt'
print('Converting images to .pt format...')
png2pt(hrpath_t, pt_pathhr, trainlisthr)
png2pt(hrpath_v, pt_pathhr, validlisthr)
png2pt(lrpath_t, pt_pathlr, trainlistlr)
png2pt(lrpath_v, pt_pathlr, validlistlr)
print('Done!')

# Data
training_data = {
	'hrpath': pt_pathhr,
	'lrpath': pt_pathlr,
	'trainlisthr': trainlisthr,
	'trainlistlr': trainlistlr,
	'validlisthr': validlisthr,
	'validlistlr': validlistlr,
}

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

def new_model_train():
    model = ABmodel() 
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=MeanAbsoluteError())
    model.fit(training_data, callbacks=[LearningRateScheduler(scheduler)], epochs=num_epochs)
    model.save("model", include_optimizer=True, save_format='tf')

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
    dts = Lambda(lambda x: tf.nn.depth_to_space(x, 3))
    with tfmot.quantization.keras.quantize_scope({'NoOpQuantizeConfig': NoOpQuantizeConfig, 'depth_to_space': dts, 'tf': tf}):
        model = tfmot.quantization.keras.quantize_apply(model)
    model.compile(optimizer=Adam(learning_rate=qat_learning_rate), loss=MeanAbsoluteError())
    model.fit(training_data, callbacks=[LearningRateScheduler(qat_scheduler)], epochs=qat_num_epochs)
    model.save("qat_model", include_optimizer=True, save_format='tf')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', default=None) # flags if we want to do quantization training, followed by the path to the model file
    args = parser.parse_args()
    if args.q == None:
        new_model_train()
    else:
        qat_train(args.q)