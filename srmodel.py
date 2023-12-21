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
import cv2
import numpy as np
import math
import shutil
import os
import pickle
import absl.logging as al
from data import Data
from tensorboardX import SummaryWriter

# disable unnecessarily verbose warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
al.set_verbosity(al.ERROR)

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

# Paths
model_path = './model'
qat_model_path = './qat_model'
eval_path = './qat_model/eval'
tflite_path = './qat_model/quantized.tflite'

# Network Architecture
# ====================

def ABmodel():
	# Input
	inputs = Input(shape=(None, None, 3))
	upsample_lambda = Lambda(lambda x: tf.concat(x, axis=3))
	upsampled = upsample_lambda([inputs]*(3**2))

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
	dts_lambda = Lambda(lambda x: tf.nn.depth_to_space(x, 3))
	x = dts_lambda(x)

	# Clip node to restrict pixel values to the range 0 to 255, this gives final output
	clip_lambda = Lambda(lambda x: K.clip(x, 0., 255.))
	outputs = clip_lambda(x)

	return Model(inputs=inputs, outputs=outputs, name='anchor_based_plain_net')

# Model Training
# ==============

class validation_callback(Callback):
	def __init__(self, validation_data, training_data, save_path, tbx, state):
		super(validation_callback, self).__init__()
		self.validation_data = validation_data
		self.training_data = training_data
		self.save_path = save_path
		self.tbx = tbx
		self.best_epoch = state['best_epoch']
		self.best_psnr = state['best_psnr']

	def on_epoch_end(self, epoch, logs):
		psnr = 0.0
		for _, (lr, hr) in enumerate(self.validation_data):
			sr = self.model(lr)
			sr_numpy = K.eval(sr)
			psnr += self.calc_psnr((sr_numpy).squeeze(), (hr).squeeze())
		psnr = round(psnr / len(self.validation_data), 4)
		loss = round(logs['loss'], 4)

		# save best status
		if psnr >= self.best_psnr:
			self.best_psnr = psnr
			self.best_epoch = epoch
			self.model.save(self.save_path, overwrite=True, include_optimizer=True, save_format='tf')
		state = {'cur_epoch': epoch, 'best_epoch': self.best_epoch, 'best_psnr': self.best_psnr}
		if not os.path.exists(self.save_path + '/state.pkl'):
			open(self.save_path + '/state.pkl', 'a').close()
		with open(self.save_path + '/state.pkl', 'wb') as f:
			pickle.dump(state, f)

		# save tensorboard
		self.tbx.add_scalar('psnr', psnr, epoch+1)
		self.tbx.add_scalar('loss', loss, epoch+1)


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

def new_model_train(log_path):
	state = {'cur_epoch': -1, 'best_epoch': -1, 'best_psnr': -1}
	model = ABmodel()
	if os.path.exists(model_path) and os.path.exists(log_path):
		tbx_writer = SummaryWriter(log_path)
		model = load_model(model_path)
		with open(model_path + '/state.pkl', 'rb') as f:
			state = pickle.load(f)
	else:
		if os.path.exists(log_path):
			shutil.rmtree(log_path)
		os.makedirs(log_path)
		if os.path.exists(model_path):
			shutil.rmtree(model_path)
		tbx_writer = SummaryWriter(log_path)
		model.compile(optimizer=Adam(learning_rate=learning_rate), loss=MeanAbsoluteError())
	model.fit(
		training_data,
		callbacks=[
			LearningRateScheduler(scheduler),
			validation_callback(validation_data, training_data, model_path, tbx_writer, state)
		],
		epochs=num_epochs,
		initial_epoch=state['cur_epoch']+1,
		workers=8
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
	if 'lambda' in layer.name:
		return tfmot.quantization.keras.quantize_annotate_layer(layer, quantize_config=NoOpQuantizeConfig())
	return layer

def qat_scheduler(epoch, lr):
	if epoch % qat_lr_reduction_interval == 0:
		lr = lr*0.5
	return lr

def qat_train(log_path):
	state = {'cur_epoch': -1, 'best_epoch': -1, 'best_psnr': -1}
	if os.path.exists(qat_model_path) and os.path.exists(log_path):
		tbx_writer = SummaryWriter(log_path)
		model = load_model(qat_model_path)
		with open(qat_model_path + '/state.pkl', 'rb') as f:
			state = pickle.load(f)
	else:
		if os.path.exists(log_path):
			shutil.rmtree(log_path)
		os.makedirs(log_path)
		tbx_writer = SummaryWriter(log_path)
		if os.path.exists(qat_model_path):
			shutil.rmtree(qat_model_path)
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
			validation_callback(validation_data, training_data, qat_model_path, tbx_writer, state)
		],
		epochs=qat_num_epochs,
		initial_epoch=state['cur_epoch']+1,
		workers=8
	)

# Model Quantization
# ==================
	
# set up representative data (one image) for quantization (min. size 360x640, 360p)
def rep_data():
	lr_img_path = 'data/DIV2K_LR_pt/0001x3.pt'
	with open(lr_img_path, 'rb') as f:
		lr_img = pickle.load(f)
	lr_img = lr_img.astype(np.float32)
	lr_img = np.expand_dims(lr_img, 0)
	if lr_img.shape[1] >=360 and lr_img.shape[2] >= 640:
		yield [lr_img[:, 0:360, 0:640, :]]

# generate tflite file from qat model (quantized, properly sized), this is the final model and will be used for evaluation and benchmarking
def gen_tflite(qat_p, tflite_p):
	tnsr_shp = [1, 360, 640, 3] # model will super resolve 360p images to 1080p (3x)
	gen_rep = rep_data
	qat_model = tf.saved_model.load(qat_p)
	conc = qat_model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
	conc.inputs[0].set_shape(tnsr_shp)
	conv = tf.lite.TFLiteConverter.from_concrete_functions([conc])
	conv.experimental_new_converter = True
	conv.experimental_new_quantizer = True
	conv.optimizations = [tf.lite.Optimize.DEFAULT]
	conv.representative_dataset = gen_rep
	conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
	conv.inference_input_type = tf.uint8
	conv.inference_output_type = tf.uint8
	tfl = conv.convert()
	with open(tflite_p, 'wb') as f:
		f.write(tfl)

def eval(tflite_p, eval_p):
	inter = tf.lite.Interpreter(model_path=tflite_p)
	total_psnr = 0.0
	for i in range(801, 901):
		print(f'Evaluating using image {i}')
		lr_p = f'data/DIV2K_LR_pt/0{i}x3.pt'
		with open(lr_p, 'rb') as f:
			lr = pickle.load(f)
		lr = np.expand_dims(lr, 0).astype(np.float32)
		lr = lr.astype(np.uint8)
		hr_p = f'data/DIV2K_HR_pt/0{i}.pt'
		with open(hr_p, 'rb') as f:
			hr = pickle.load(f)
		hr = np.expand_dims(hr, 0).astype(np.float32)
		inter.resize_tensor_input(inter.get_input_details()[0]['index'], lr.shape)
		inter.allocate_tensors()
		inter.set_tensor(inter.get_input_details()[0]['index'], lr)
		inter.invoke()
		sr = inter.get_tensor(inter.get_output_details()[0]['index'])
		sr = np.clip(sr, 0, 255)
		_, h, w, _ = sr.shape
		out_png = os.path.join(eval_p, '{:04d}x3.png'.format(i))
		cv2.imwrite(out_png, cv2.cvtColor(sr.squeeze().astype(np.uint8), cv2.COLOR_RGB2BGR))
		mse = np.mean((sr[:, 1:h-1, 1:w-1, :].astype(np.float32) - hr[:, 1:h-1, 1:w-1, :].astype(np.float32)) ** 2)
		psnr =  20. * math.log10(255. / math.sqrt(mse))
		total_psnr += psnr
	print(f'Average PSNR: {total_psnr / 100}')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-q', action='store_true', default=False) # flag if we want to do quantization training, requires a model to have been created
	parser.add_argument('-g' , action='store_true', default=False) # flag to generate tflite file by quantizing the qat model
	args = parser.parse_args()
	if args.g:
		print('Quantization: Generating TFLite file')
		gen_tflite(qat_model_path, tflite_path)
		eval(tflite_path, eval_path)
	elif args.q:
		print('Quantization Aware Training')
		qat_train('log/qat_model_tbx')
	else:
		print('Preliminary Training')
		new_model_train('log/model_tbx')
