import torch.nn as nn
from torchvision import models
from torch.nn import Module

def get_model() -> Module:
    """
    Loads a MobileNetV2 model with pretrained weights, model chosen for efficiency and accuracy.
    Changes the final layer to output a single value (the angle).
    Returns the modified model.
    """
    model = models.mobilenet_v2(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.last_channel, 1)
    return model

