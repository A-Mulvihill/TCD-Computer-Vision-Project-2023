import argparse
import tensorflow as tf
import keras.backend as K
from keras.callbacks import Callback, LearningRateScheduler
from keras.layers import Conv2D, Input, Lambda, Add
from keras.models import Model
from keras.initializers import he_normal
from keras.optimizers import Adam
from keras.losses import MeanAbsoluteError
from keras.models import load_model, clone_model
import tensorflow_model_optimization as tfmot
import numpy as np
import math
import os.path as osp
from data import Data

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
training_data = Data('train')
validation_data = Data('valid')

# Model Paths
model_path = "./model"
qat_model_path = "./qat_model"

# Network Architecture
# ====================

def ABmodel():
	# Input
	inputs = Input(shape=(None, None, 3))
	upsampled = tf.concat([inputs]*(3**2), axis=3)

	# Shallow Feature Extraction, single 3x3 Convolution layer followed by ReLU
	x = Conv2D(num_kernels, conv_kernel_size, activation='relu', kernel_initializer=he_normal(), padding='same')(inputs)

	# Deep Feature Extraction, 5 layers of 3x3 Convolution layers paired with ReLUs
	for _ in range(num_deep_feature_layers):
		x = Conv2D(num_kernels, conv_kernel_size, activation='relu', kernel_initializer=he_normal(), padding='same')(x)

	# Transfer features to HR image space, one convolution layer (Assume from the paper's figure it is 3x3 conv)
	x = Conv2D(3*(3**2), conv_kernel_size, kernel_initializer=he_normal(), padding='same')(x) # This layer needs to have an output with the same shape as upsampled

	# Apply ABRL
	x = Add()([upsampled, x])

	# Reshuffle Pixels, labelled as depth to space in paper's figure
	x = tf.nn.depth_to_space(x, 3)

	# Clip node to restrict pixel values to the range 0 to 255, this gives final output
	outputs = K.clip(x, 0., 255.)

	return Model(inputs=inputs, outputs=outputs, name='anchor_based_plain_net')

# Model Training
# ==============

class validation_callback(Callback):
	def __init__(self, validation_data, training_data, save_path):
		super(validation_callback, self).__init__()
		self.validation_data = validation_data
		self.training_data = training_data
		self.save_path = save_path
		self.best_psnr = -1

	def on_epoch_end(self, epoch, logs):
		psnr = 0.0
		for _, (lr, hr) in enumerate(self.validation_data):
			sr = self.model(lr)
			sr_numpy = K.eval(sr)
			psnr += self.calc_psnr((sr_numpy).squeeze(), (hr).squeeze())
		psnr = round(psnr / len(self.validation_data), 4)

		# save best status
		if psnr >= self.best_psnr:
			self.best_psnr = psnr
			self.model.save(self.save_path, overwrite=True, include_optimizer=True, save_format='tf')


	def calc_psnr(self, y, y_target):
		h, w, c = y.shape
		y = np.clip(np.round(y), 0, 255).astype(np.float32)
		y_target = np.clip(np.round(y_target), 0, 255).astype(np.float32)

		# crop 1
		y_cropped = y[1:h-1, 1:w-1, :]
		y_target_cropped = y_target[1:h-1, 1:w-1, :]
		
		mse = np.mean((y_cropped - y_target_cropped) ** 2)
		if mse == 0:
			return 100
		return 20. * math.log10(255. / math.sqrt(mse))

def scheduler(epoch, lr):
	if epoch % lr_reduction_interval == 0:
		lr = lr*0.5
	return lr

def new_model_train():
	model = ABmodel()
	if osp.exists(model_path):
		model = load_model(model_path)
	else:
		model.compile(optimizer=Adam(learning_rate=learning_rate), loss=MeanAbsoluteError())
	model.fit(
		training_data,
		callbacks=[
			LearningRateScheduler(scheduler),
			validation_callback(validation_data, training_data, model_path)
		],
		epochs=qat_num_epochs
	)

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

def qat_train():
	model = load_model(model_path)
	model = clone_model(model, clone_function=qat_helper)
	model = tfmot.quantization.keras.quantize_annotate_model(model)
	dts = Lambda(lambda x: tf.nn.depth_to_space(x, 3))
	with tfmot.quantization.keras.quantize_scope({'NoOpQuantizeConfig': NoOpQuantizeConfig, 'depth_to_space': dts, 'tf': tf}):
		model = tfmot.quantization.keras.quantize_apply(model)
	model.compile(optimizer=Adam(learning_rate=qat_learning_rate), loss=MeanAbsoluteError())
	model.fit(
		training_data,
		callbacks=[
			LearningRateScheduler(qat_scheduler),
			validation_callback(validation_data, training_data, qat_model_path)
		],
		epochs=qat_num_epochs
	)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-q', action='store_true', default=False) # flags if we want to do quantization training, requires a model to have been created
	args = parser.parse_args()
	if args.q:
		qat_train()
	else:
		new_model_train()