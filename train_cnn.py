import os
import json
import prettytable as pt
from collections import Counter
import argparse
from pathlib import Path
import math
import time

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

# Create parser
def parse_args():
    parser = argparse.ArgumentParser(description='Process the task type.')

    parser.add_argument('--task', type=str, default='object', choices=['action', 'object'],
                        help='Specify the task type: "action" or "object".')
    
    parser.add_argument('--explanation_type', type=str, default="multimodal", 
                        choices=["multimodal", "visual", "none", "text"],
                        help="Specify the type of explanation-guided learning (default: multimodal)")
    
    parser.add_argument('--model_type', type=str, default="resnet50", choices=["resnet50", "resnet101", "resnet34"],
                        help='Specify the type of CNN model (default: resnet50).')
    
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Specify the number of epochs for training (default: 3).')
    
    parser.add_argument('--transformation', type=str, default="GRADIA", choices=["GRADIA", "HAICS"],
                        help='Specify the type of transformation for the visual explanation (default: GRADIA).')
    
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='Specify the batch size (default: 8).')
    
    parser.add_argument('--att_weight', type=float, default=0.1,
                        help='Weight of the attention loss term in the loss function (default: 1).')
    
    parser.add_argument('--exp_weight', type=float, default=1,
                        help='Weight of the explanation loss term in the loss function (default: 1).')
    
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Specify the learning rate (default: 1e-4).')

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
    root_dir = Path('Datasets/Action_Classification/')
elif args.task == "object":
    root_dir = Path('Datasets/Object_Classification/')

def custom_collate_fn(batch):
    human_input_lst = []
    gpt_output_lst = []
    image_lst = []
    visual_exp_lst = []
    class_id_lst = []
    ans_lst = []
    for item in batch:
        human_input, gpt_output, image_path, visual_exp_path, class_id, ans = item

        human_input_lst.append(human_input)
        gpt_output_lst.append(gpt_output)

        image = Image.open(image_path).convert('RGB')

        if visual_exp_path:
            visual_exp = np.load(visual_exp_path)
            if visual_exp.ndim == 3 and visual_exp.shape[0] == 3:  # Check if it's a 3-channel image
                visual_exp = visual_exp.transpose(1, 2, 0)  # Reorder dimensions to (height, width, channels)
                if visual_exp.dtype == np.float64:
                    visual_exp = (visual_exp * 255).astype(np.uint8)  # Assuming the data is in range [0, 1]
            
            visual_exp = Image.fromarray(visual_exp.astype('uint8'))  # Assume it's grayscale
            visual_exp = visual_exp.convert('RGB')
        else:
            visual_exp = Image.new('RGB', image.size, (0, 0, 0)) # Dummy image of zeros
        
        image = transform_image(image)
        image_lst.append(image)

        visual_exp = transform_visualexp(visual_exp)
        visual_exp_lst.append(visual_exp)

        class_id = torch.tensor(class_id, dtype=torch.long)
        class_id_lst.append(class_id)

        ans_lst.append(ans)
    
    images = torch.stack(image_lst)
    visual_exps = torch.stack(visual_exp_lst)
    class_ids = torch.tensor(class_id_lst)

    return human_input_lst, gpt_output_lst, images, visual_exps, class_ids, ans_lst


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

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn)

model = MEGL_CNN(
    explanation_type=args.explanation_type,
    num_classes=num_classes,
    cnn_type=args.model_type
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=args.lr)

def lr_lambda(epoch):
    warmup_epochs = 10
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    else:
        return 0.5 * (1 + math.cos((epoch - warmup_epochs) * math.pi / (50 - warmup_epochs)))

scheduler = LambdaLR(optimizer, lr_lambda)

print("-"*10 + "Training Starts" + "-"*10)

for epoch in range(args.num_epochs):
    model.train()
    running_loss = 0.0
    running_pred_loss = 0.0
    running_att_loss = 0.0

    grad_cam = GradCam(
        model=model.cnn_net,
        feature_module=model.cnn_net.layer4,
        target_layer_names=["1"],
        use_cuda=True
    )

    for i, data in enumerate(train_loader):
        start_time = time.time()
        human_input, gpt_output, images, visual_exps, class_ids, ans = data
        images = images.to(device)
        class_ids = class_ids.to(device)
        y_pred_label, y_pred_text = model(images)
        pred_loss = criterion(y_pred_label, class_ids)
        
        if args.explanation_type in ["multimodal", "visual"]:
            # att_map, _ = grad_cam.get_attention_map(images, class_ids, norm="ReLU")
            if visual_exps.cpu().numpy().any():
                # att_map_lst = [
                #     grad_cam.get_attention_map(torch.unsqueeze(image, 0), label, norm="ReLU")[0] 
                #     for image, label in zip(images, class_ids)
                # ]
                att_maps, _ = grad_cam.get_attention_map(images, class_ids, norm="ReLU")
                visual_exps_resized = normalize_and_resize(visual_exps).to(device)
                # att_maps = torch.stack(att_map_lst)
                att_loss = F.l1_loss(att_maps, visual_exps_resized)
                visual_exps_resized_trans = visual_exp_transform(visual_exps_resized, args.transformation)
                att_loss += cal_trans_att_loss(att_maps, visual_exps_resized_trans, args.transformation)
            else:
                att_loss = torch.zeros(1, dtype=torch.float32).to(device)
        else:
            att_loss = torch.zeros(1, dtype=torch.float32).to(device)
        
        scheduler.step()

        loss = pred_loss + args.att_weight * att_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update running totals
        running_loss += loss.item()
        running_pred_loss += pred_loss.item()
        running_att_loss += att_loss.item()

        if (i + 1) % 5 == 0:
            end_time = time.time()
            print(f'[{epoch + 1} / {args.num_epochs}, {i + 1} / {len(train_loader)}] loss: {running_loss / 100:.3f}, '
                  f'pred_loss: {running_pred_loss / 100:.3f}, '
                  f'att_loss: {running_att_loss / 100:.3f}, time:{end_time - start_time:.3f}')
            running_loss = 0.0
            running_pred_loss = 0.0
            running_att_loss = 0.0

    if epoch + 1 == args.num_epochs:
        break

    model.eval()
    correct = 0
    total = 0
    total_label_tuned = []
    total_pred_tuned = []
    total_outputs_tuned = []
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Fine-tuned Model Evaluation"):
            human_input, gpt_output, images, visual_exps, class_ids, ans = data
            images = images.to(device)
            labels = class_ids.to(device)
            outputs, y_pred_text = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_label_tuned.extend(labels.cpu().tolist())
            total_pred_tuned.extend(predicted.cpu().tolist())
            total_outputs_tuned.extend(outputs.cpu().tolist())

    total_outputs_tuned = np.array(total_outputs_tuned)
    total_label_tuned = np.array(total_label_tuned)

    test_accuracy_tuned = accuracy_score(total_label_tuned, total_pred_tuned)
    test_recall_tuned_micro = recall_score(total_label_tuned, total_pred_tuned, average="micro")
    test_precision_tuned_micro = precision_score(total_label_tuned, total_pred_tuned, average="micro")
    test_f1_tuned_micro = f1_score(total_label_tuned, total_pred_tuned, average="micro")

    soft_max_outputs_tuned = torch.tensor(total_outputs_tuned)
    soft_max_outputs_tuned = F.softmax(soft_max_outputs_tuned)

    test_auc_tuned_micro = roc_auc_score(total_label_tuned, soft_max_outputs_tuned.numpy(), 
                                        average="micro", multi_class="ovr")

    test_recall_tuned_macro = recall_score(total_label_tuned, total_pred_tuned, average="macro")
    test_precision_tuned_macro = precision_score(total_label_tuned, total_pred_tuned, average="macro")
    test_f1_tuned_macro = f1_score(total_label_tuned, total_pred_tuned, average="macro")
    test_auc_tuned_macro = roc_auc_score(total_label_tuned, soft_max_outputs_tuned.numpy(), 
                                        average="macro", multi_class="ovr")

    print("-"*10 + f"Epoch {epoch + 1}" + "-"*10)
    tb = pt.PrettyTable()
    tb.field_names = ["", "Accuracy", "Recall", "Precision", "F1", "AUC"]
    tb.add_row(
        ["Micro",test_accuracy_tuned, test_recall_tuned_micro, 
        test_precision_tuned_micro, test_f1_tuned_micro, test_auc_tuned_micro]
    )
    tb.add_row(
        ['Macro', test_accuracy_tuned, test_recall_tuned_macro, 
        test_precision_tuned_macro, test_f1_tuned_macro, test_auc_tuned_macro]
    )
    print(tb)

# Save the model
torch.save(model.state_dict(), "model_resnet50.pth")

model.eval()
correct = 0
total = 0
total_label_tuned = []
total_pred_tuned = []
total_outputs_tuned = []
with torch.no_grad():
    for data in tqdm(test_loader, desc="Fine-tuned Model Evaluation"):
        human_input, gpt_output, images, visual_exps, class_ids, ans = data
        images = images.to(device)
        labels = class_ids.to(device)
        outputs, y_pred_text = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        total_label_tuned.extend(labels.cpu().tolist())
        total_pred_tuned.extend(predicted.cpu().tolist())
        total_outputs_tuned.extend(outputs.cpu().tolist())

total_outputs_tuned = np.array(total_outputs_tuned)
total_label_tuned = np.array(total_label_tuned)

test_accuracy_tuned = accuracy_score(total_label_tuned, total_pred_tuned)
test_recall_tuned_micro = recall_score(total_label_tuned, total_pred_tuned, average="micro")
test_precision_tuned_micro = precision_score(total_label_tuned, total_pred_tuned, average="micro")
test_f1_tuned_micro = f1_score(total_label_tuned, total_pred_tuned, average="micro")

soft_max_outputs_tuned = torch.tensor(total_outputs_tuned)
soft_max_outputs_tuned = F.softmax(soft_max_outputs_tuned)

test_auc_tuned_micro = roc_auc_score(total_label_tuned, soft_max_outputs_tuned.numpy(), 
                                     average="micro", multi_class="ovr")

test_recall_tuned_macro = recall_score(total_label_tuned, total_pred_tuned, average="macro")
test_precision_tuned_macro = precision_score(total_label_tuned, total_pred_tuned, average="macro")
test_f1_tuned_macro = f1_score(total_label_tuned, total_pred_tuned, average="macro")
test_auc_tuned_macro = roc_auc_score(total_label_tuned, soft_max_outputs_tuned.numpy(), 
                                     average="macro", multi_class="ovr")

print("-"*10 + "After Fine-tuning" + "-"*10)
tb = pt.PrettyTable()
tb.field_names = ["", "Accuracy", "Recall", "Precision", "F1", "AUC"]
tb.add_row(
    ["Micro",test_accuracy_tuned, test_recall_tuned_micro, 
     test_precision_tuned_micro, test_f1_tuned_micro, test_auc_tuned_micro]
)
tb.add_row(
    ['Macro', test_accuracy_tuned, test_recall_tuned_macro, 
     test_precision_tuned_macro, test_f1_tuned_macro, test_auc_tuned_macro]
)
print(tb)
