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
            clip_model_name_or_path:str="llava-hf/llava-1.5-7b-hf",
            llava_model_name_or_path:str="llava-hf/llava-1.5-7b-hf",
            hidden_size:int=1024
    ) -> None:
        super().__init__()
        self.explanation_type = explanation_type
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        # Initilization of CNN
        if cnn_type == "resnet50":
            self.cnn_net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif cnn_type == "resnet101":
            self.cnn_net = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        elif cnn_type == "resnet34":
            self.cnn_net = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.cnn_net.fc = nn.Linear(self.cnn_net.fc.in_features, num_classes)

        # TODO: LLaVA Init
        if self.explanation_type in ['text', 'multimodal']:
            self.llava_model = LlavaForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=llava_model_name_or_path,
                torch_dtype=torch.bfloat16
            )
            self.processor = AutoProcessor.from_pretrained(llava_model_name_or_path)
            self.embedding_shape = self.llava_model.language_model.model.embed_tokens.embedding_dim
            self.projector = nn.Sequential(
                nn.Linear(self.num_classes, self.hidden_size, bias=True),
                nn.ReLU(),
                nn.Linear(self.hidden_size, 4096, bias=True)
            )
        else:
            self.processor, self.llava_model = None, None
        
    def forward(
            self, 
            images:torch.Tensor,
            llava_inputs:dict
    ) -> torch.Tensor:
        if self.explanation_type in ['none', 'visual']:
            y_label = self.cnn_net(images)
            y_text = None
        
        # TODO: CNN + LLaVA code
        elif self.explanation_type in ['text', 'multimodal']:
            # CNN
            y_label = self.cnn_net(images)

            # LLaVA
            # 1. Extract the input
            input_ids = llava_inputs["input_ids"]
            attention_mask = llava_inputs["attention_mask"]
            pixel_values = llava_inputs["pixel_values"]

            # 2. Obtain the input embeddings
            inputs_embeds = self.llava_model.get_input_embeddings()(input_ids)

            # 3. Merge the images and the text
            image_outputs = self.llava_model.vision_tower(
                pixel_values, output_hidden_states=True
            )
            selected_image_feature = image_outputs.hidden_states[-2]
            del image_outputs # to save memory
            selected_image_feature = selected_image_feature[:, 1:]
            image_features = self.llava_model.multi_modal_projector(selected_image_feature)
            labels = None
            inputs_embeds, attention_mask, labels, position_ids = self.llava_model._merge_input_ids_with_image_features(
                    image_features, inputs_embeds, input_ids, attention_mask, labels
            )

            y_text = self.llava_model(inputs_embeds=inputs_embeds)

        return y_label, y_text

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MEGL_CNN(explanation_type="multimodal", num_classes=10).to(device)
    print(model)
    print(model.cnn_net)
