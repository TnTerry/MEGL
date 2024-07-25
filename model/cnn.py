import torch
import torch.nn as nn
from torchvision import transforms, models
from MEGL.utils.utils import *
from MEGL.utils.gradcam import GradCam

class MEGL_CNN(nn.Module):
    def __init__(self, explanation_type:str, num_classes:int) -> None:
        super().__init__()
        self.explanation_type = explanation_type
        self.cnn_net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.cnn_net.fc = nn.Linear(self.cnn_net.fc.in_features, num_classes)
        
    def forward(self, inputs:torch.Tensor) -> torch.Tensor:
        if self.explanation_type in ["none", "visual"]:
            y = self.cnn_net(inputs)
        
        # TODO: CNN + LLaVA code

        return y