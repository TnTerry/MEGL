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

# Create the parser
parser = argparse.ArgumentParser(description='Process the task type.')

parser.add_argument('--task', type=str, default='action', choices=['action', 'object'],
                    help='Specify the task type: "action" or "object".')
parser.add_argument('--num_epochs', type=int, default=4,
                    help='Specify the number of epochs for training (default: 4).')
parser.add_argument('--att_weight', type=float, default=1,
                    help='Weight of the attention loss term in the loss function (default: 1).')

# Parse the command-line arguments
args = parser.parse_args()

# Get the directory path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Set the current working directory to the script directory
os.chdir(script_dir)
print("Current Working Directory: ", os.getcwd())

# Assuming data is defined properly
def load_json_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

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

# Assuming CustomDataset is defined somewhere (in utils.py)
dataset = MMESDataset(data, root_dir, transform_image, transform_visualexp)

# Adjust the num_classes based on the actual dataset
class_id_lst = [int(dataset[i]['class_id']) for i in range(len(dataset))]
class_id_cnt = Counter(class_id_lst)
print(class_id_cnt)
num_classes = len(class_id_cnt)

# Split the dataset into train and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_indices, test_indices = train_test_split(
    range(len(dataset)),
    test_size = 0.2
)

train_dataset = Subset(data, train_indices)
test_dataset = Subset(data, test_indices)
sr_sampler_train = SubsetRandomSampler(train_indices)
sr_sampler_test = SubsetRandomSampler(test_indices)


# DataLoaders for train and test datasets
train_loader = DataLoader(dataset, batch_size=16, 
                          sampler=sr_sampler_train, collate_fn=custom_collate_fn)
test_loader = DataLoader(dataset, batch_size=16, 
                         sampler=sr_sampler_test, collate_fn=custom_collate_fn)

llava_model_name = "show_model/model001"
llava_processor = LlavaProcessor.from_pretrained(llava_model_name)
llava_model = LlavaForConditionalGeneration.from_pretrained(
    llava_model_name, device_map="cuda:0", torch_dtype=torch.bfloat16
)

