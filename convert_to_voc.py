import os
import numpy as np

cmap = [ 128, 0, 0] #aeroplane in VOC


IMAGE_TRAIN_PATH = "dataset/img_dir/train/"
MASK_TRAIN_PATH = "dataset/ann_dir/train"

IMAGE_VAL_PATH = "dataset/img_dir/val/"
MASK_VAL_PATH = "dataset/ann_dir/val/"

if not os.path.isdir('dataset2'):
        os.makedirs('dataset2')

def sorted_fns(dir):
    return sorted(os.listdir(dir), key=lambda x: x.split('.')[0])


mask_train_ids = np.array([os.path.join(MASK_TRAIN_PATH, x) for x in sorted_fns(MASK_TRAIN_PATH)])
img_train_ids = np.array([os.path.join(IMAGE_TRAIN_PATH, x) for x in sorted_fns(IMAGE_TRAIN_PATH)])

mask_val_ids = np.array([os.path.join(MASK_VAL_PATH, x) for x in sorted_fns(MASK_VAL_PATH)])
img_val_ids = np.array([os.path.join(IMAGE_VAL_PATH, x) for x in sorted_fns(IMAGE_VAL_PATH)])

size = len(img_train_ids)
sizev = len(img_val_ids)
print(size)
print(sizev)