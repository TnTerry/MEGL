import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset, SubsetRandomSampler
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
torch.autograd.set_detect_anomaly(True)
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from accelerate import Accelerator, DistributedType

from PIL import Image
import numpy as np
import os
import json
import cv2
import prettytable as pt
from collections import Counter
import argparse
from pathlib import Path

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


from utils import *
from gradcam import GradCam

# Create parser
def parse_args():
    parser = argparse.ArgumentParser(description='Process the task type.')
    parser.add_argument('--task', type=str, default='action', choices=['action', 'object'],
                        help='Specify the task type: "action" or "object".')
    parser.add_argument('--explanation_type', type=str, default="multimodal", 
                        choices=["multimodal", "attention", "none", "text"],
                        help="Specify the type of explanation-guided learning (default: multimodal)")
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='Specify the number of epochs for training (default: 1).')
    parser.add_argument('--att_weight', type=float, default=1,
                        help='Weight of the attention loss term in the loss function (default: 1).')
    parser.add_argument('--exp_weight', type=float, default=1,
                        help='Weight of the explanation loss term in the loss function (default: 1).')
    args = parser.parse_args()
    return args

# Get the directory path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Set the current working directory to the script directory
os.chdir(script_dir)
print("Current Working Directory: ", os.getcwd())

# Assuming data is defined properly
def load_json_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

args = parse_args()
if args.task == "action":
    root_dir = 'Datasets/Action_Classification/'  # Adjust as necessary to point to the correct root directory
    data = load_json_data("Datasets/Action_Classification/exp_annotation.json")
    id_map = load_json_data("Datasets/Action_Classification/label_class_id_mapping.json")
elif args.task == "object":
    data = load_json_data("Datasets/Object_Classification/exp_annotation.json")
    root_dir = 'Datasets/Object_Classification/'  # Adjust as necessary to point to the correct root directory
    id_map = load_json_data("Datasets/Object_Classification/label_class_id_mapping.json")
else:
    raise ValueError("Invalid task specified. Expected 'action' or 'object'.")

# Load a pre-trained ResNet-50 model and modify it for your number of classes
num_classes = len(id_map)  # Update this to your actual number of classes

def custom_collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    visual_exps = torch.stack([item['visual_exp'] for item in batch])
    class_ids = torch.tensor([item['class_id'] for item in batch])
    exps = [item['exp'] for item in batch]  # List of lists

    # If you're using tokenized indices for text and want to pad:
    # exps_padded = pad_sequence([torch.tensor(exp) for sublist in exps for exp in sublist], 
    #                            batch_first=True, padding_value=0)

    return {'image': images, 'visual_exp': visual_exps, 'class_id': class_ids, 'exp': exps}

transform_image = transforms.Compose([
                  transforms.Resize((256, 256)),
                  transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]) 
        
transform_visualexp = transforms.Compose([
                      transforms.Resize((256, 256)),
                      transforms.ToTensor()
                      ]) 