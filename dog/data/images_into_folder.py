import os.path
import sys
import pandas as pd
from tqdm import tqdm
import shutil

this_dir = os.path.dirname(__file__)
sys.path.insert(0, this_dir + '/..')

from config import cfg 


def creat_folder(lables):
	all_breeds = set(lables['breed'])
	for breed in tqdm(all_breeds):
		path = os.path.join(cfg.Path.DATA_ROOT, breed)
		os.makedirs(path)

def put_images_into_folders(labels):
	for name in tqdm(os.listdir(cfg.Path.DATA_ROOT)):
		if not os.path.isdir(os.path.join(cfg.Path.DATA_ROOT, name)):
			index = name.split('.')[0]
			label = labels[labels['id']==index]['breed'].values[0]
			shutil.move(os.path.join(cfg.Path.DATA_ROOT, name), os.path.join(cfg.Path.DATA_ROOT, label))


if __name__ == '__main__':
	labels = pd.read_csv(cfg.Path.LABELS)
	#creat_folder(labels)
	put_images_into_folders(labels)

