import os
import json
import prettytable as pt
from collections import Counter
import argparse
from pathlib import Path
import math
import time
import logging

from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import wandb

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset, SubsetRandomSampler
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
torch.autograd.set_detect_anomaly(True)
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from accelerate import Accelerator, DistributedType

from model.utils.utils import *
from model.utils.data import *
from model.utils.gradcam import GradCam
from model.cnn import MEGL_CNN

def parse_args():
    parser = argparse.ArgumentParser(description='Process the task type.')

    parser.add_argument('--task', type=str, default='object', choices=['action', 'object'],
                        help='Task type: "action" or "object".')
    
    parser.add_argument('--explanation_type', type=str, default="multimodal", 
                        choices=["multimodal", "visual", "none", "text"],
                        help="Type of explanation-guided learning (default: multimodal)")
    
    parser.add_argument('--model_type', type=str, default="resnet50", choices=["resnet50", "resnet101", "resnet34"],
                        help='Type of CNN model (default: resnet50).')
    
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of epochs for training (default: 3).')
    
    parser.add_argument('--transformation', type=str, default="GRADIA", choices=["GRADIA", "HAICS"],
                        help='Type of transformation for the visual explanation (default: GRADIA).')
    
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='Batch size (default: 8).')
    
    parser.add_argument('--att_weight', type=float, default=0.1,
                        help='Weight of the attention loss term in the loss function (default: 1).')
    
    parser.add_argument('--exp_weight', type=float, default=1,
                        help='Weight of the explanation loss term in the loss function (default: 1).')
    
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4).')
    
    parser.add_argument("--lmm_model_name_or_path", type=str, default="llava-hf/llava-1.5-7b-hf",
                        help="LMM model path")
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )

    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )

    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")

    args = parser.parse_args()
    return args

args = parse_args()

# Get the directory path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Set the current working directory to the script directory
os.chdir(script_dir)
print("Current Working Directory: ", os.getcwd())

# Assuming data is defined properly
def load_json_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

accelerator_log_kwargs = {}

if args.with_tracking:
    accelerator_log_kwargs["log_with"] = args.report_to
    accelerator_log_kwargs["project_dir"] = args.output_dir

accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)


if args.task == "action":
    root_dir = Path('Datasets/Action_Classification/')
elif args.task == "object":
    root_dir = Path('Datasets/Object_Classification/')

transform_image = transforms.Compose([
                  transforms.Resize((256, 256)),
                  transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]) 
        
transform_visualexp = transforms.Compose([
                      transforms.Resize((256, 256)),
                      transforms.ToTensor()
                      ])

dataset = MEGLDataset(root_dir)

# Adjust the num_classes based on the actual dataset
class_id_lst = [int(dataset[i][4]) for i in range(len(dataset))]
class_id_cnt = Counter(class_id_lst)
print("Acutal Dataset Label:\n", class_id_cnt)
num_classes = len(class_id_cnt)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

processor = AutoProcessor.from_pretrained(args.lmm_model_name_or_path)
megl_collator = TrainMEGLCollator(processor, -100)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=megl_collator)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=megl_collator)

model = MEGL_CNN(
    explanation_type=args.explanation_type,
    num_classes=num_classes,
    cnn_type=args.model_type,
    llava_model_name_or_path=args.lmm_model_name_or_path
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

