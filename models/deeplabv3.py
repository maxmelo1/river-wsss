""" DeepLabv3 Model. Não daria tempo de implementar ela manualmente, portanto vou usar o modelo pronto e mudar apenas o classificador para classificação binária. """
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models


import segmentation_models_pytorch as smp


def DeepLabv3(outputchannels=1):

    model = smp.DeepLabV3Plus(classes=1, 
    activation=None)
    return model