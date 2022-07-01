# A Comparative Study of River Segmentation Based on Deep Learning Methods

## Abstract

The monitoring of river regions represents an important aspect for the expansion of knowledge about ecology and also about hydrological processes. This paper tackles the problem of river segmentation using aerial images using both Weakly an Fully Supervised Semantic Segmentation methods. U-Net, Deeplab v3+, Attention-Unet, UNet++ and PuzzleCAM were evaluated. For Fully Supervised methods, UNet++ achieved best results with $96.86$ points of IoU. PuzzleCAM performed $52.15$ IoU points, as a Weakly Supervised method.

## Train requirements

In this project, it was used:
- Python 3.8, PyTorch 1.7.0, and more in requirements.txt
- CUDA 10.1, cuDNN 7.6.5
- 1x Nvidia P100

## Dataset
It was used a manually annotated dataset of rivers in the Santo Antonio Basin. It will not be publicly available yet. As soon as there is an accepted publication, it will be made available.

## Train FSSS model
```bash
python train.py train.py --bs 8 \
--epochs 100 \
--width 256 \
--height 256 \
--model $model_name
```

A simplified version is available in test.sh. Run with:
```bash
sh train.sh 
```

The available models are: ['Unet', 'AttUnet', 'Deeplab', 'UnetPlus']

## Test FSSS model
To test and evaluate the trained model, run:
```bash
python test.py --bs 8 \
--epochs 100 \
--width 256 \
--height 256 \
--model $model_name
```

Or

```bash
sh test.sh 
```

## Colab notebook

Google colab was used to train and test the FSSS models. The notebook used is available [here](https://colab.research.google.com/drive/1G0g-J_p5ZU1F5JL78HshG0tDNV4sknxI?usp=sharing).


## Quantitative results

|Model| Acc | Precision | Recall | F1 | IoU
|---| ---:| ---:| ---:| ---:| ---:|
U-Net |99.0| 98.93 | 54.67 | 69.93 |86.36|
Deeplab v3+| 99.09 | 91.11 | 70.45 | 78.84 | 91.11|
Attention U-Net | 98.96 | 97.95 | 52.16| 67.66| 85.19|
UNet++ | 99.29| 100.0| 79.73| 88.76| 96.86

## Train and test WSSS model
The PuzzleCAM was trained as explained by the author [PuzzleCAM Github](https://github.com/OFRIN/PuzzleCAM).

## Quantitative results

|Model|  mIoU
|---| ---:|
PuzzleCAM |52.15|


## Colab notebook

Google colab was used to train and test the PuzzleCAM model. The notebook used is available [here](https://colab.research.google.com/drive/146E6TFXqe0W8qMAopBmIQP1pJYQlsmO9?usp=sharing).