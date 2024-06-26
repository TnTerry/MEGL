import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
import os
import json
import cv2

def normalize_and_resize(tensor:torch.Tensor) -> torch.Tensor:
    # Ensure the tensor is a floating point data type for accurate division
    tensor = tensor.float()
    
    # Normalize the tensor to [0, 1] range
    min_val = tensor.min()
    max_val = tensor.max()
    if max_val > min_val:
        # Normalize if there is a range
        tensor = (tensor - min_val) / (max_val - min_val)
    else:
        # Handle the case where all values are the same
        tensor = torch.zeros_like(tensor)

    # Resize the tensor using adaptive average pooling
    resized_tensor = F.adaptive_avg_pool2d(tensor, (8, 8))
    resized_tensor = resized_tensor.mean(dim=0)

    return resized_tensor

def BF_solver(X, Y):
    epsilon = 1e-4

    with torch.no_grad():
        x = torch.flatten(X)
        y = torch.flatten(Y)
        g_idx = (y<0).nonzero(as_tuple=True)[0]
        le_idx = (y>0).nonzero(as_tuple=True)[0]
        len_g = len(g_idx)
        len_le = len(le_idx)
        a = 0
        a_ct = 0.0
        for idx in g_idx:
            v = x[idx] + epsilon # to avoid miss the constraint itself
            v_ct = 0.0
            for c_idx in g_idx:
                v_ct += (v>x[c_idx]).float()/len_g
            for c_idx in le_idx:
                v_ct += (v<=x[c_idx]).float()/len_le
            if v_ct>a_ct:
                a = v
                a_ct = v_ct

        for idx in le_idx:
            v = x[idx]
            v_ct = 0.0
            for c_idx in g_idx:
                v_ct += (v>x[c_idx]).float()/len_g
            for c_idx in le_idx:
                v_ct += (v<=x[c_idx]).float()/len_le
            if v_ct>a_ct:
                a = v
                a_ct = v_ct

    return torch.tensor([a]).cuda()

def visual_exp_transform(visual_exp_map:torch.Tensor, trans_type:str=None) -> torch.Tensor:
    '''
    Transform visual explanation with HAICS, GRADIA or Gaussian
    '''
    if not trans_type or trans_type == "HAICS" or trans_type == "GRADIA":
        return visual_exp_map
    
    if trans_type == "Gaussian":
        visual_exp_map_pos = np.maximum(visual_exp_map.cpu().numpy(), 0)
        visual_exp_map_trans = cv2.GaussianBlur(visual_exp_map_pos, (3,3), 0) # Gaussian Blur
        visual_exp_map_trans = visual_exp_map_trans / (np.max(visual_exp_map_trans)+1e-6)
    
    return torch.from_numpy(visual_exp_map_trans).cuda()

# class MMESDataset(Dataset):
#     def __init__(self, data, root_dir, transform_image, transform_visualexp):
#         self.data = data
#         self.root_dir = root_dir
#         self.transform_image = transform_image
#         self.transform_visualexp = transform_visualexp

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         keys = list(self.data.keys())
#         item = self.data[keys[idx]]

#         # Load image
#         img_path = os.path.join(self.root_dir, item['image_path'])
#         image = Image.open(img_path).convert('RGB')

#         # Handle visual_exp
#         visual_exp_path = item['visual_exp']
#         if visual_exp_path and os.path.isfile(os.path.join(self.root_dir, visual_exp_path)):
#             visual_exp = np.load(os.path.join(self.root_dir, visual_exp_path))
            
#             # Convert the array to the correct shape and type
#             if visual_exp.ndim == 3 and visual_exp.shape[0] == 3:  # Check if it's a 3-channel image
#                 visual_exp = visual_exp.transpose(1, 2, 0)  # Reorder dimensions to (height, width, channels)
#                 if visual_exp.dtype == np.float64:
#                     visual_exp = (visual_exp * 255).astype(np.uint8)  # Assuming the data is in range [0, 1]
     
#             visual_exp = Image.fromarray(visual_exp.astype('uint8'))  # Assume it's grayscale
#             visual_exp = visual_exp.convert('RGB')  # Convert grayscale to RGB if necessary
#         else:
#             # If no npy file, create a dummy image of zeros
#             visual_exp = Image.new('RGB', image.size, (0, 0, 0))

#         # Apply transformations
#         image = self.transform_image(image)
#         visual_exp = self.transform_visualexp(visual_exp)  # Ensure this transformation includes a ToTensor()

#         class_id = torch.tensor(item['class_id'], dtype=torch.long)
#         exp_texts = item['exp']  # No transformation applied, handle as list of strings

#         return {
#             'image': image,
#             'visual_exp': visual_exp,
#             'class_id': class_id,
#             'exp': exp_texts  # Return the list of explanations
#         }

class MMESDataset(Dataset):
    def __init__(self, data, root_dir, transform_image, transform_visualexp):
        self.data = data
        self.keys = list(data.keys())
        self.root_dir = root_dir
        self.transform_image = transform_image
        self.transform_visualexp = transform_visualexp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        key = self.keys[idx]  # Use pre-cached keys
        item = self.data[key]

        # Load image
        img_path = os.path.join(self.root_dir, item['image_path'])
        image = Image.open(img_path).convert('RGB')

        # Handle visual_exp
        visual_exp_path = item['visual_exp']
        if visual_exp_path and os.path.isfile(os.path.join(self.root_dir, visual_exp_path)):
            visual_exp = np.load(os.path.join(self.root_dir, visual_exp_path))
            
            # Convert the array to the correct shape and type
            if visual_exp.ndim == 3 and visual_exp.shape[0] == 3:  # Check if it's a 3-channel image
                visual_exp = visual_exp.transpose(1, 2, 0)  # Reorder dimensions to (height, width, channels)
                if visual_exp.dtype == np.float64:
                    visual_exp = (visual_exp * 255).astype(np.uint8)  # Assuming the data is in range [0, 1]
     
            visual_exp = Image.fromarray(visual_exp.astype('uint8'))  # Assume it's grayscale
            visual_exp = visual_exp.convert('RGB')  # Convert grayscale to RGB if necessary
        else:
            # If no npy file, create a dummy image of zeros
            visual_exp = Image.new('RGB', image.size, (0, 0, 0))

        # Apply transformations
        image = self.transform_image(image)
        visual_exp = self.transform_visualexp(visual_exp)  # Ensure this transformation includes a ToTensor()

        class_id = torch.tensor(item['class_id'], dtype=torch.long)
        exp_texts = item['exp']  # No transformation applied, handle as list of strings

        return {
            'image': image,
            'visual_exp': visual_exp,
            'class_id': class_id,
            'exp': exp_texts  # Return the list of explanations
        }
        