from easydict import EasyDict as edict
import os
cfg = edict()

# Path
cfg.Path = edict()
cfg.Path.DATA_ROOT = 'E:/data/dog_breed/image'
cfg.Path.TEST = 'E:/data/dog_breed/test'
cfg.Path.LABELS = 'E:/data/dog_breed/labels.csv'


# Arch
cfg.Arch = edict()

cfg.Arch.EPOCHS = 100
cfg.Arch.LR = 0.0001
cfg.Arch.TrainEx = 10222

