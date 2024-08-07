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

from .utils.gradcam import GradCam
from .utils.utils import *

class MEGL_CNN(nn.Module):
    def __init__(
            self, 
            explanation_type:str, 
            num_classes:int, 
            cnn_type:str="resnet50",
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
            labels = llava_inputs['labels']

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


class MEGL_CNN_multimodal(nn.Module):
    def __init__(
            self, 
            explanation_type:str, 
            num_classes:int, 
            cnn_type:str="resnet50",
            llava_model_name_or_path:str="llava-hf/llava-1.5-7b-hf",
            hidden_size:int=1024,
            transformation_type = "GRADIA"
    ) -> None:
        super().__init__()
        self.explanation_type = explanation_type
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.transformation_type = transformation_type

        # Initilization of CNN
        if cnn_type == "resnet50":
            self.cnn_net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif cnn_type == "resnet101":
            self.cnn_net = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        elif cnn_type == "resnet34":
            self.cnn_net = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.cnn_net.fc = nn.Linear(self.cnn_net.fc.in_features, num_classes)

        self.grad_cam = GradCam(
            model=self.model.cnn_net,
            feature_module=model.cnn_net.layer4,
            target_layer_names=["1"],
            use_cuda=True
        )

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
            llava_inputs:dict
    ) -> torch.Tensor:
        images = llava_inputs['images']
        visual_exps = llava_inputs['visual_exps']
        if self.explanation_type in ['none', 'visual']:
            y_label = self.cnn_net(images)
            att_loss = None
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
            labels = llava_inputs['labels']
            class_ids = llava_inputs['class_ids']

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
            inputs_embeds, attention_mask, labels, position_ids = self.llava_model._merge_input_ids_with_image_features(
                    image_features, inputs_embeds, input_ids, attention_mask, labels
            )

            inputs_embeds += self.projector(y_label)

            y_text = self.llava_model(inputs_embeds=inputs_embeds, 
                                      labels=labels,
                                      attention_mask=attention_mask)

        grad_cam = GradCam(
            model=model.cnn_net,
            feature_module=model.cnn_net.layer4,
            target_layer_names=["1"],
            use_cuda=True
        )

        if visual_exps.cpu().numpy().any():
            att_maps, _ = grad_cam.get_attention_map(images, class_ids, norm="ReLU")
            visual_exps_resized = normalize_and_resize(visual_exps).to(device)
            # att_maps = torch.stack(att_map_lst)
            att_loss = F.l1_loss(att_maps, visual_exps_resized)
            visual_exps_resized_trans = visual_exp_transform(visual_exps_resized, self.transformation_type)
            att_loss += cal_trans_att_loss(att_maps, visual_exps_resized_trans, self.transformation_type)
        else:
            att_loss = torch.zeros(1, dtype=torch.float32).to(device)

        return y_label, y_text, att_loss


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MEGL_CNN_multimodal(explanation_type="multimodal", num_classes=10).to(device)
    grad_cam = GradCam(
        model=model.cnn_net,
        feature_module=model.cnn_net.layer4,
        target_layer_names=["1"],
        use_cuda=True
    )
