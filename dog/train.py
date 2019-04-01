from model.base_net import model
from config import cfg 
from data.data import train_generator

epochs = cfg.Arch.EPOCHS 
learning_rate = cfg.Arch.LR 
decay = learning_rate / epochs
adam = Adam(lr=cfg.Arch.LR)
model.compile(
	loss='categorical_crossentropy',
	optimizer=adam,
	metrics=['accuracy'])

model.fit_generator(
	train_generator,
	steps_per_epoch=cfg.Arch.TrainEx // 32,
	epochs=epochs,
	)