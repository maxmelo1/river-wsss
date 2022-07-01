import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image
from torchmetrics import JaccardIndex#, F1Score, Precision, Recall

from custom_dataset import CustomDataset

import numpy as np

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = CustomDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = CustomDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def show_result_images(loader, model, device="cuda", use_wb=True ):
    results = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            transform = T.ToPILImage()

            for im, mask, pred in zip(x, y.squeeze(), preds.squeeze()):
                results.append([transform(im), transform(pred), transform(mask)])

            #results.append(transform(preds.squeeze()), transform(y.squeeze()))

    return results


def check_accuracy(loader, model, device="cuda", use_wb=True, eps=1e-7 ):
    num_correct = 0
    num_pixels = 0
    n_samples = 0
    model.eval()

    jaccard = JaccardIndex(num_classes=2).to(device)

    iou_torch = 0.0
    f1_torch = 0.0
    prec_torch = 0.0
    rec_torch = 0.0

    with torch.no_grad():
        for x, y in loader:
            # print(len(x))
            n_samples = len(x)
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            
            preds = torch.sigmoid(model(x))

            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

            tp = torch.sum(preds * y)  
            fp = torch.sum(preds * (1 - y))  
            fn = torch.sum((1 - preds) * y)  
            tn = torch.sum((1 - preds) * (1 - y)) 
            
            y = y.int()

            iou_torch += jaccard(preds, y).to(device)
            precision = (tp + eps) / (tp + fp + eps)
            recall = (tp + eps) / (tp + fn + eps)
            f1 = (2*precision*recall)/(precision+recall).mean()

            
            prec_torch+= precision
            rec_torch += recall
            f1_torch  += f1

            

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    
    print(f"IOU : {iou_torch/n_samples*100:.2f} (Batch = {n_samples})")
    print(f"Precision : {prec_torch/n_samples*100:.2f} ")
    print(f"Recall : {rec_torch/n_samples*100:.2f} ")
    print(f"F1 : {f1_torch/n_samples*100:.2f} ")
    if use_wb:
        import wandb

        wandb.log({"iou": iou_torch/n_samples})
    model.train()

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()