import os.path
import sys
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten

from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

this_dir = os.path.dirname(__file__)
sys.path.insert(0, this_dir + '/..')

from config import cfg
from data.data import train_generator

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(220, 220, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(120, activation='softmax'))

model.summary()

epochs = cfg.Arch.EPOCHS 
learning_rate = cfg.Arch.LR 
decay = learning_rate / epochs
adam = Adam(lr=cfg.Arch.LR, )
model.compile(
	loss='categorical_crossentropy',
	optimizer=adam,
	metrics=['accuracy'])

model.fit_generator(
	train_generator,
	steps_per_epoch=cfg.Arch.TrainEx // 32,
	epochs=epochs,
	)