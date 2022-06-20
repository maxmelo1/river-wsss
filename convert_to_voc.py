import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json

pix_cls = (128, 0, 0) #aeroplane in VOC


IMAGE_TRAIN_PATH = "dataset/img_dir/train/"
MASK_TRAIN_PATH = "dataset/ann_dir/train"

IMAGE_VAL_PATH = "dataset/img_dir/val/"
MASK_VAL_PATH = "dataset/ann_dir/val/"

if not os.path.isdir('dataset2'):
        os.makedirs('dataset2')

if not os.path.isdir('dataset2/JPEGImages'):
        os.makedirs('dataset2/JPEGImages')
        os.makedirs('dataset2/SegmentationClass')
        os.makedirs('data')

def sorted_fns(dir):
    return sorted(os.listdir(dir), key=lambda x: x.split('.')[0])

def save_files(img_name, mask_name):
    image = Image.open(img_name).convert("RGB")
    mask = np.array(Image.open(mask_name).convert("L"), dtype=np.float32)

    
    new_mask = np.zeros( (mask.shape[0], mask.shape[1], 3), dtype='uint8' )

    new_mask[mask == 1.0] = pix_cls

    new_img_name = img_name.split('/')[3].split('.')[0]
    image.save("dataset2/JPEGImages/"+new_img_name+".jpg")

    new_mask_name = mask_name.split('/')[3].split('.')[0]
    mask = Image.fromarray(new_mask).convert('RGB')
    mask.save('dataset2/SegmentationClass/'+new_mask_name+".png")

def count_river(mask_ids):
    c=0
    for mask_train in mask_ids:
        mask = np.array(Image.open(mask_train).convert("L"), dtype=np.float32)
        c += np.sum(mask) > 0
    
    return c

def save_dict(train_size, val_size):
    print(type(train_size))
    voc_dict = {"train":{
                        "river": int(train_size)
                    }, 
                "validation":{
                        "river": int(val_size)
                    }, 
                "classes": 1,
                "class_names":["river"],
                "class_dic":{"river": 0},
                "color_dict": {
                    "background": [
                        0,
                        0,
                        0
                    ],
                    "river": [
                        128,
                        0,
                        0
                    ]}
    }

    with open('data/VOC_2012.json', 'w') as f:
        json.dump(voc_dict, f)


mask_train_ids = np.array([os.path.join(MASK_TRAIN_PATH, x) for x in sorted_fns(MASK_TRAIN_PATH)])
img_train_ids = np.array([os.path.join(IMAGE_TRAIN_PATH, x) for x in sorted_fns(IMAGE_TRAIN_PATH)])

mask_val_ids = np.array([os.path.join(MASK_VAL_PATH, x) for x in sorted_fns(MASK_VAL_PATH)])
img_val_ids = np.array([os.path.join(IMAGE_VAL_PATH, x) for x in sorted_fns(IMAGE_VAL_PATH)])

size = len(img_train_ids)
sizev = len(img_val_ids)
# print(img_train_ids)
# print(sizev)

if os.path.exists('data/train.txt'):
    os.remove('data/train.txt')
if os.path.exists('data/val.txt'):
    os.remove('data/val.txt')
if os.path.exists('data/train_aug.txt'):
    os.remove('data/train_aug.txt')


fval = open('data/val.txt', 'a')
ftrain = open('data/train.txt', 'a')
ftrain_aug = open('data/train_aug.txt', 'a') 
#TODO implementar o boundaries para train_aug

# for img_train, mask_train in zip(img_train_ids, mask_train_ids):
#     ftrain.write(img_train.split('/')[3].split('.')[0]+'\n')

#     save_files(img_train, mask_train)

# for img_val , mask_val in zip(img_val_ids, mask_val_ids):
#     fval.write(img_val.split('/')[3].split('.')[0]+'\n')

#     save_files(img_train, mask_val)

fval.close()
ftrain.close()
ftrain_aug.close()


save_dict(count_river(mask_train_ids), count_river(mask_val_ids))