
import torch

LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 128  # 1280 originally
IMAGE_WIDTH = 128  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "dataset/img_dir/train"
TRAIN_MASK_DIR = "dataset/ann_dir/train"
VAL_IMG_DIR = "dataset/img_dir/val"
VAL_MASK_DIR = "dataset/ann_dir/val"
VALID_MODELS = ['Unet', 'AttUnet', 'Deeplab', 'UnetPlus', 'Puzzle-CAM']
MODEL = 'Unet'