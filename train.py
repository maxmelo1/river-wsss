import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import argparse

from configs.params import *

from models.unet import UNET
from models.attention_unet import AttUNET
from models.smp_models import DeepLabv3, UnetPlus
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

import matplotlib.pyplot as plt
import numpy as np

import wandb

wandb.init(project="bacia_sto_antonio", entity="maxmelo")

def train_fn(loader, model, optimizer, loss_fn, scaler, epoch, scheduler):
    loop = tqdm(loader, unit="batch")

    for batch_idx, (data, targets) in enumerate(loop):
        loop.set_description(f"Epoch {epoch}")

        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        wandb.log({"loss": loss, "lr": scheduler.get_last_lr()[0]})

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
    return loss.item()

def train():
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

    if MODEL == 'Unet':
        model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    elif MODEL == 'AttUnet':
        model = AttUNET(in_channels=3, out_channels=1).to(DEVICE)
    elif MODEL == 'Deeplab':
        model = DeepLabv3(outputchannels=1).to(DEVICE)
    elif MODEL == 'UnetPlus':
        model = UnetPlus(output_channels=1).to(DEVICE)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

    train_loader, val_loader = get_loaders(
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

    

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    # el = next(iter(val_loader))

    # print(np.shape(X))
    # print(np.shape(y))
    
    # plt.imshow(X.permute(1,2,0) )
    # plt.show()
    # plt.imshow(y, cmap='gray' )
    # plt.show()

    check_accuracy(val_loader, model, device=DEVICE)
    
    scaler = torch.cuda.amp.GradScaler()

    wandb.config = {
        "learning_rate": LEARNING_RATE,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE
    }

    best_loss = 99999.99

    for epoch in range(NUM_EPOCHS):
        loss = train_fn(train_loader, model, optimizer, criterion, scaler, epoch, scheduler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        if loss< best_loss:
            save_checkpoint(checkpoint, filename="results/"+MODEL+"/last_checkpoint.pth.tar")
            loss = best_loss

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        # save_predictions_as_imgs(
        #     val_loader, model, folder="saved_images/", device=DEVICE
        # )
        scheduler.step()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='WSSS Training Function')

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

    train()