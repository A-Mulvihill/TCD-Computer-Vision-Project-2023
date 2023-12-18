import numpy as np
import tensorflow as tf
import cv2
import random
import os
import os.path as osp
import pickle

class Data(tf.keras.utils.Sequence): # tf.keras.utils.Sequence is a base class for fitting to a sequence of data, such as a dataset
	# split: 'train' or 'valid'
	def __init__(self, split):
		# default
		self.patch_size = 64
		self.batch_size = 16
		self.scale = 3
		self.split = split
		self.pt_pathhr = 'data/DIV2K_HR_pt'
		self.pt_pathlr = 'data/DIV2K_LR_pt'
		self.hrpath = 'data/DIV2K_train_HR'
		self.lrpath = 'data/DIV2K_train_LR_bicubic/X3'

		if split == 'train':
			print('Converting training images to .pt format...')
			self.flip = True
			self.rot = True
			self.enlarge_times = 20
			self.hrlist = ['%04d.pt' % i for i in range(1, 801)] # '0001.pt' to '0800.pt'
			self.lrlist = ['%04dx3.pt' % i for i in range(1, 801)] # '0001x3.pt' to '0800x3.pt'

		else: # split == 'valid'
			print('Converting validation images to .pt format...')
			self.flip = None
			self.rot = None
			self.enlarge_times = 1
			self.hrlist = ['%04d.pt' % i for i in range(801, 901)] # '0801.pt' to '0900.pt'
			self.lrlist = ['%04dx3.pt' % i for i in range(801, 901)] # '0801x3.pt' to '0900x3.pt'

		print('converting hr images...')
		self.png2pt(self.hrpath, self.pt_pathhr, self.hrlist)
		print('converting lr images...')
		self.png2pt(self.lrpath, self.pt_pathlr, self.lrlist)
		print('Done!')

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

	# functions called internally by model.fit() and model.evaluate()
	# ===============================================================
	def shuffle(self):
		random.shuffle(self.hrlist) # We will only use hrlist for indexing as we can only shuffle one list at a time

	def __len__(self):
		if self.split == 'train':
			return int(len(self.hrlist) * self.enlarge_times / self.batch_size)
		else:
			return len(self.hrlist)
		
	def __getitem__(self, idx):
		start = (idx * self.batch_size) 
		end = start + self.batch_size
		if self.split == 'train':
			lr_batch = np.zeros((self.batch_size, self.patch_size, self.patch_size, 3), dtype=np.float32)
			hr_batch = np.zeros((self.batch_size, self.patch_size * self.scale, self.patch_size * self.scale, 3), dtype=np.float32)
		for i in range(start, end):
			lr, hr = self.get_image_pair(i % len(self.hrlist))
			lr_batch[i - start] = lr
			hr_batch[i - start] = hr
		else:
			lr, hr = self.get_image_pair(idx)
			lr_batch, hr_batch = np.expand_dims(lr, 0), np.expand_dims(hr, 0)
		return (lr_batch).astype(np.float32), (hr_batch).astype(np.float32)
	
	# functions called internally by __getitem__()
	# ============================================
	def get_image_pair(self, idx):
		full_hrpath = osp.join(self.hrpath, self.hrlist[idx])
		base, ext = osp.splitext(self.hrlist[idx])
		lr_basename = base + 'x{}'.format(self.scale) + '.pt'
		full_lrpath = osp.join(self.lrpath, lr_basename)
		# Read the images
		with open(full_hrpath, 'rb') as f:
			hr = pickle.load(f)
		with open(full_lrpath, 'rb') as f:
			lr = pickle.load(f)
		if self.split == 'train':
			lr_patch, hr_patch = self.get_patch(lr, hr, self.patch_size, self.scale)
			lr, hr = self.augment(lr_patch, hr_patch, self.flip, self.rot)
		return lr, hr
	
	def get_patch(self, lr, hr, ps, scale):
		lr_h, lr_w = lr.shape[:2]
		lr_x = random.randint(0, lr_w - ps)
		lr_y = random.randint(0, lr_h - ps)
		hr_x = lr_x * scale
		hr_y = lr_y * scale
		lr_patch = lr[lr_y : lr_y+ps, lr_x : lr_x+ps, :]
		hr_patch = hr[hr_y : hr_y+ps*scale, hr_x : hr_x+ps*scale, :]
		return lr_patch, hr_patch
	
	def augment(self, lr, hr, flip, rot):
		hflip = flip and random.random() < 0.5
		vflip = flip and random.random() < 0.5
		rot90 = rot and random.random() < 0.5
		if hflip:
			lr = np.ascontiguousarray(lr[:, ::-1, :])
			hr = np.ascontiguousarray(hr[:, ::-1, :])
		if vflip:
			lr = np.ascontiguousarray(lr[::-1, :, :])
			hr = np.ascontiguousarray(hr[::-1, :, :])
		if rot90:
			lr = lr.transpose(1, 0, 2)
			hr = hr.transpose(1, 0, 2)
			return lr, hr
