import os
import sys
import random
import csv


import argparse

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from configs.params import *
from models.unet import UNET
from models.attention_unet import AttUNET
from models.smp_models import DeepLabv3, UnetPlus#, DeepLabV3Plus
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    show_result_images,
)

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from PIL import Image

from matplotlib.pyplot import imshow

COLORS = {'RIVER': [128, 0, 0], 'BG': [35, 2 ,38]}

def test(filename, outpath):
    assert filename != "", "invalid filename given"
    assert outpath != "", "invalid output path given"

    if not  os.path.exists(outpath):
        os.mkdir(outpath)

    if MODEL == 'Unet':
        model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    elif MODEL == 'AttUnet':
        model = AttUNET(in_channels=3, out_channels=1).to(DEVICE)
    elif MODEL == 'Deeplab':
        model = DeepLabv3(output_channels=1).to(DEVICE)
    elif MODEL == 'UnetPlus':
        model = UnetPlus(output_channels=1).to(DEVICE)


    load_checkpoint(torch.load(filename), model)

    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    _, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    check_accuracy(val_loader, model, device=DEVICE, use_wb=False)

    print('\n\n\n')

    results = show_result_images(val_loader, model, device=DEVICE, use_wb=False)

    print('Saving predictions...')
    x = []
    y = []
    for i, r in enumerate(results):
        print(f'Sample {i}')
        im = r[0]

        mask = np.zeros((IMAGE_WIDTH, IMAGE_HEIGHT, 3), dtype='uint8')
        pred = np.zeros((IMAGE_WIDTH, IMAGE_HEIGHT, 3), dtype='uint8')

        rpred = np.array(r[1])
        pred[:,:] = COLORS['BG']
        pred[rpred>0] = COLORS['RIVER']

        rmask = np.array(r[2])
        mask[:,:] = COLORS['BG']
        mask[rmask>0] = COLORS['RIVER']

        pred = Image.fromarray(pred)
        pred.save(outpath+str(i)+"_pred.png")

        mask = Image.fromarray(mask)
        mask.save(outpath+str(i)+"_mask.png")

        im.save(outpath+str(i)+".png")
    
    print('Done saving!')
        
        



if __name__ == '__main__':



    parser = argparse.ArgumentParser(description='WSSS Testing Function')

    parser.add_argument('--bs', help='batch size', type=int, required=False)
    parser.add_argument('--epochs', help='max epochs', type=int, required=False)
    parser.add_argument('--height', help='input image height', type=int, required=False)
    parser.add_argument('--width', help='input image width', type=int, required=False)
    parser.add_argument('--model', help='model name', type=str, required=False)
    
    args = parser.parse_args()
    
    BATCH_SIZE      = args.bs
    NUM_EPOCHS      = args.epochs
    IMAGE_HEIGHT    = args.height
    IMAGE_WIDTH     = args.width

    #default: Unet
    if args.model in VALID_MODELS:
        MODEL           = args.model
    
    print(f'SELECTED MODEL: {MODEL}')

    test("results/"+MODEL+"/last_checkpoint.pth.tar", "results/"+MODEL+"/images/")