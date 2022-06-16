""" DeepLabv3 Model. Não daria tempo de implementar ela manualmente, portanto vou usar o modelo pronto e mudar apenas o classificador para classificação binária. """
from torch import classes
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models


import segmentation_models_pytorch as smp


def DeepLabv3(output_channels=1):
    model = smp.DeepLabV3Plus(classes=output_channels, 
    activation=None)
    
    return model


def UnetPlus(output_channels=1):
    model = smp.UnetPlusPlus(classes=output_channels, decoder_attention_type='scse',
    activation=None)
    
    return model