#coding=utf-8
from matplotlib import pyplot as plt 
from PIL import Image
import keras
import os.path
import sys
from keras.preprocessing.image import ImageDataGenerator

this_dir = os.path.dirname(__file__)
sys.path.insert(0, this_dir + '/..')
from config import cfg 

print(cfg.Path.DATA_ROOT)

batch_size = 32
train_datagen =ImageDataGenerator(
	rescale=1. / 255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
	cfg.Path.DATA_ROOT,
	target_size=(220, 220),
	batch_size=batch_size,
	class_mode='categorical')

