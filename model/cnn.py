import sys
import os

import torch
import torch.nn as nn
from torchvision import transforms, models
from transformers import (
    LlavaForConditionalGeneration, 
    AutoProcessor
)

# from model.utils.utils import *
# from model.utils.gradcam import GradCam

class MEGL_CNN(nn.Module):
    def __init__(
            self, 
            explanation_type:str, 
            num_classes:int, 
            cnn_type:str="resnet50",
            clip_model_name_or_path:str="model\LLaVA\clip-vit-large-patch14-336",
            llava_model_name_or_path:str="model\LLaVA\llava-v1.5-7b",
    ) -> None:
        super().__init__()
        self.explanation_type = explanation_type
        if cnn_type == "resnet50":
            self.cnn_net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif cnn_type == "resnet101":
            self.cnn_net = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        elif cnn_type == "resnet34":
            self.cnn_net = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.cnn_net.fc = nn.Linear(self.cnn_net.fc.in_features, num_classes)
        # TODO: LLaVA Init
        # self.clip_model = AutoModel.from_pretrained(clip_model_name_or_path, device_map=self.device)
        if self.explanation_type in ['text', 'multimodal']:
            self.llava_model = LlavaForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=llava_model_name_or_path,
                torch_dtype=torch.bfloat16
            )
            self.processor = AutoProcessor.from_pretrained(llava_model_name_or_path)
        else:
            self.processor, self.llava_model = None, None
        
    def forward(self, inputs:torch.Tensor) -> torch.Tensor:
        if self.explanation_type in ['none', 'visual']:
            y_label = self.cnn_net(inputs)
            y_text = None
        
        # TODO: CNN + LLaVA code
        elif self.explanation_type in ['text', 'multimodal']:
            y_label = self.cnn_net(inputs)
            y_text = None

        return y_label, y_text
    

if __name__ == "__main__":
    model = MEGL_CNN(explanation_type="multimodal", num_classes=10)
    print(model)
    print(model.cnn_net)
