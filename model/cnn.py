import sys
import os

import torch
import torch.nn as nn
from torchvision import transforms, models

from model.utils.utils import *
from model.utils.gradcam import GradCam

class MEGL_CNN(nn.Module):
    def __init__(self, explanation_type:str, num_classes:int) -> None:
        super().__init__()
        self.explanation_type = explanation_type
        self.cnn_net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.cnn_net.fc = nn.Linear(self.cnn_net.fc.in_features, num_classes)
        # TODO: LLaVA Init
        
    def forward(self, inputs:torch.Tensor) -> torch.Tensor:
        if self.explanation_type in ['none', 'visual']:
            y_label = self.cnn_net(inputs)
            y_text = None
        
        # TODO: CNN + LLaVA code
        elif self.explanation_type in ['text', 'multimodal']:
            y_label = self.cnn_net(inputs)
            y_text = None

        return y_label, y_text
