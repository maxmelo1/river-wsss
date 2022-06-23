import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
import cv2
from pascal_voc_writer import Writer


palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]


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


def show_image(image):
    img2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(img2)
    plt.show()
   

def findBoundingBoxes(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, im = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY)
    contours, hierarchy  = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    res_img = image.copy()
    img = cv2.drawContours(res_img, contours, -1, (0,255,0), 2)
    
    box_list = []

    for contour in contours:
        (x,y, w,h) = cv2.boundingRect(contour)
        cv2.rectangle(res_img, (x,y), (w+x, h+y), (255,0,0), 2)
        box_list.append((x,y, w,h))

    #uncoment if you want to preview the contours and bounding boxes over the mask.
    #show_image(res_img)
    return box_list

def save_files(img_name, mask_name):
    cmap = np.array([[[0,0,0]],[[128,0,0]]], dtype=np.uint8) #aeroplane in VOC

    image = Image.open(img_name).convert("RGB")
    mask = np.array(Image.open(mask_name).convert('L'))    

    #mask = np.array(mask.convert("L"))[:, :, np.newaxis]
    
    # new_mask = np.zeros( (mask.shape[0], mask.shape[1], 3), dtype='uint8' )
    # new_mask[mask == 1.0] = pix_cls

    # new_mask = np.dot(mask == 0, cmap[0])
    # new_mask += np.dot(mask == 1, cmap[1])
    new_mask = np.array(mask, copy=True)
    new_mask[new_mask>0] = 1

    new_img_name = img_name.split('/')[3].split('.')[0]
    image.save("dataset2/JPEGImages/"+new_img_name+".jpg")

    new_mask_name = mask_name.split('/')[3].split('.')[0]
    #mask = Image.fromarray(new_mask).convert('RGB')
    #mask = Image.fromarray(new_mask.astype(np.uint8))

    new_mask = Image.fromarray(new_mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    new_mask.save('dataset2/SegmentationClass/'+new_mask_name+".png")

    writer = Writer(path="dataset2/SegmentationClass/"+new_mask_name+".png", width=np.shape(new_mask)[0], height=np.shape(new_mask)[1], segmented=1)

    cv_image = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
    blist = findBoundingBoxes(cv_image)
    #print(blist, new_mask_name+".png")

    for (x, y, w, h) in blist:
        writer.addObject('river', x, y, x+w, y+h)
    writer.save("dataset2/Annotations/"+new_mask_name+".xml")

    

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

for img_train, mask_train in zip(img_train_ids, mask_train_ids):
    ftrain.write(img_train.split('/')[3].split('.')[0]+'\n')

    save_files(img_train, mask_train)

for img_val , mask_val in zip(img_val_ids, mask_val_ids):
    fval.write(img_val.split('/')[3].split('.')[0]+'\n')

    save_files(img_val, mask_val)

fval.close()
ftrain.close()
ftrain_aug.close()


save_dict(count_river(mask_train_ids), count_river(mask_val_ids))