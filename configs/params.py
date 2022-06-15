
import torch

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 128  # 1280 originally
IMAGE_WIDTH = 128  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "dataset/train_images/"
TRAIN_MASK_DIR = "dataset/train_masks/"
VAL_IMG_DIR = "dataset/val_images/"
VAL_MASK_DIR = "dataset/val_masks/"