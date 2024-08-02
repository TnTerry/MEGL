import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from pathlib import Path
from PIL import Image
import numpy as np
import os
import json
import cv2

BCE_criterion = nn.BCELoss()
l1_criterion = nn.L1Loss(reduction="none")

# def normalize_and_resize(tensor:torch.Tensor) -> torch.Tensor:
#     # Ensure the tensor is a floating point data type for accurate division
#     tensor = tensor.float()
    
#     # Normalize the tensor to [0, 1] range
#     min_val = tensor.min()
#     max_val = tensor.max()
#     if max_val > min_val:
#         # Normalize if there is a range
#         tensor = (tensor - min_val) / (max_val - min_val)
#     else:
#         # Handle the case where all values are the same
#         tensor = torch.zeros_like(tensor)

#     # Resize the tensor using adaptive average pooling
#     resized_tensor = F.adaptive_avg_pool2d(tensor, (8, 8))
#     resized_tensor = resized_tensor.mean(dim=0)

#     return resized_tensor

def normalize_and_resize(tensor: torch.Tensor) -> torch.Tensor:
    # Ensure the tensor is a floating point data type for accurate division
    tensor = tensor.float()
    
    # If the input tensor has a batch dimension
    if tensor.dim() == 4:
        batch_size, channels, height, width = tensor.size()
        
        # Normalize each sample in the batch
        min_val = tensor.view(batch_size, channels, -1).min(dim=2, keepdim=True)[0]
        max_val = tensor.view(batch_size, channels, -1).max(dim=2, keepdim=True)[0]
        
        if (max_val > min_val).all():
            tensor = (tensor - min_val) / (max_val - min_val)
        else:
            tensor = torch.zeros_like(tensor)
        
        # Resize each sample in the batch
        resized_tensor = F.adaptive_avg_pool2d(tensor, (8, 8))
        
        # Take the mean across the channel dimension
        resized_tensor = resized_tensor.mean(dim=1)
    else:
        # Single sample case (no batch dimension)
        min_val = tensor.min()
        max_val = tensor.max()
        
        if max_val > min_val:
            tensor = (tensor - min_val) / (max_val - min_val)
        else:
            tensor = torch.zeros_like(tensor)
        
        resized_tensor = F.adaptive_avg_pool2d(tensor, (8, 8))
        resized_tensor = resized_tensor.mean(dim=0)

    return resized_tensor


# def BF_solver(X, Y):
#     epsilon = 1e-4

#     with torch.no_grad():
#         x = torch.flatten(X)
#         y = torch.flatten(Y)
#         g_idx = (y<0).nonzero(as_tuple=True)[0]
#         le_idx = (y>0).nonzero(as_tuple=True)[0]
#         len_g = len(g_idx)
#         len_le = len(le_idx)
#         a = 0
#         a_ct = 0.0
#         for idx in g_idx:
#             v = x[idx] + epsilon # to avoid miss the constraint itself
#             v_ct = 0.0
#             for c_idx in g_idx:
#                 v_ct += (v>x[c_idx]).float()/len_g
#             for c_idx in le_idx:
#                 v_ct += (v<=x[c_idx]).float()/len_le
#             if v_ct>a_ct:
#                 a = v
#                 a_ct = v_ct

#         for idx in le_idx:
#             v = x[idx]
#             v_ct = 0.0
#             for c_idx in g_idx:
#                 v_ct += (v>x[c_idx]).float()/len_g
#             for c_idx in le_idx:
#                 v_ct += (v<=x[c_idx]).float()/len_le
#             if v_ct>a_ct:
#                 a = v
#                 a_ct = v_ct

#     return torch.tensor([a]).cuda()

# def BF_solver(X, Y):
#     epsilon = 1e-4

#     with torch.no_grad():
#         # Flatten the tensors along the last three dimensions (height, width, channels)
#         x = X.view(X.size(0), -1)
#         y = Y.view(Y.size(0), -1)
        
#         g_idx = (y < 0)
#         le_idx = (y > 0)
        
#         # Initialize a and a_ct
#         a = torch.zeros(X.size(0), device=X.device)
#         a_ct = torch.zeros(X.size(0), device=X.device)

#         # Compute the counts of g_idx and le_idx
#         len_g = g_idx.sum(dim=1, keepdim=True).float()
#         len_le = le_idx.sum(dim=1, keepdim=True).float()

#         # Avoid division by zero
#         len_g[len_g == 0] = 1.0
#         len_le[len_le == 0] = 1.0

#         # Compute v and v_ct for g_idx
#         v_g = x + epsilon
#         v_ct_g = torch.zeros_like(v_g)
#         v_ct_g[g_idx] = torch.cumsum((v_g.unsqueeze(2) > x.unsqueeze(1)).float(), dim=2)[:, :, 0] / len_g
#         v_ct_g[le_idx] = torch.cumsum((v_g.unsqueeze(2) <= x.unsqueeze(1)).float(), dim=2)[:, :, 0] / len_le
        
#         max_v_ct_g, idx_g = v_ct_g.max(dim=1)
#         a[max_v_ct_g > a_ct] = v_g.gather(1, idx_g.unsqueeze(1)).squeeze()[max_v_ct_g > a_ct]
#         a_ct[max_v_ct_g > a_ct] = max_v_ct_g[max_v_ct_g > a_ct]

#         # Compute v and v_ct for le_idx
#         v_le = x
#         v_ct_le = torch.zeros_like(v_le)
#         v_ct_le[g_idx] = torch.cumsum((v_le.unsqueeze(2) > x.unsqueeze(1)).float(), dim=2)[:, :, 0] / len_g
#         v_ct_le[le_idx] = torch.cumsum((v_le.unsqueeze(2) <= x.unsqueeze(1)).float(), dim=2)[:, :, 0] / len_le
        
#         max_v_ct_le, idx_le = v_ct_le.max(dim=1)
#         a[max_v_ct_le > a_ct] = v_le.gather(1, idx_le.unsqueeze(1)).squeeze()[max_v_ct_le > a_ct]
#         a_ct[max_v_ct_le > a_ct] = max_v_ct_le[max_v_ct_le > a_ct]

#     return a.unsqueeze(1)


# def visual_exp_transform(visual_exp_map:torch.Tensor, trans_type:str=None) -> torch.Tensor:
#     '''
#     Transform visual explanation with HAICS, GRADIA or Gaussian
#     '''
#     if not trans_type or trans_type == "HAICS" or trans_type == "GRADIA":
#         return visual_exp_map
    
#     if trans_type == "Gaussian":
#         visual_exp_map_pos = np.maximum(visual_exp_map.cpu().numpy(), 0)
#         visual_exp_map_trans = cv2.GaussianBlur(visual_exp_map_pos, (3,3), 0) # Gaussian Blur
#         visual_exp_map_trans = visual_exp_map_trans / (np.max(visual_exp_map_trans)+1e-6)
    
#     return torch.from_numpy(visual_exp_map_trans).cuda()


def visual_exp_transform(visual_exp_map: torch.Tensor, trans_type: str = None) -> torch.Tensor:
    '''
    Transform visual explanation with HAICS, GRADIA or Gaussian
    '''
    if not trans_type or trans_type == "HAICS" or trans_type == "GRADIA":
        return visual_exp_map

    if trans_type == "Gaussian":
        # Ensure the tensor is positive
        visual_exp_map_pos = torch.relu(visual_exp_map)
        
        # Apply Gaussian blur using convolution
        kernel = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]], device=visual_exp_map.device)
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, 3, 3).repeat(visual_exp_map.size(1), 1, 1, 1)
        
        visual_exp_map_trans = F.conv2d(visual_exp_map_pos, kernel, padding=1, groups=visual_exp_map.size(1))
        
        # Normalize the result
        max_vals = visual_exp_map_trans.view(visual_exp_map_trans.size(0), -1).max(dim=1, keepdim=True)[0]
        visual_exp_map_trans = visual_exp_map_trans / (max_vals.view(-1, 1, 1, 1) + 1e-6)
    
    return visual_exp_map_trans


# def cal_trans_att_loss(att_map, visual_exp_trans, trans_type=None):
#     '''
#     Define the loss function for the attention map transformation
#     '''
#     if not trans_type:
#         trans_att_loss = 0
    
#     if trans_type == "Gaussian":
#         a = BF_solver(att_map, visual_exp_trans)
#         temp1 = torch.tanh(5*(att_map - a))
#         temp_loss = F.l1_loss(temp1, visual_exp_trans, reduction="mean")
#         temp_size = (visual_exp_trans != 0).float()
#         eff_loss = torch.sum(temp_loss * temp_size) / torch.sum(temp_size)
#         trans_att_loss = torch.relu(torch.mean(eff_loss) - 0)

#         # att_map_labels_trans = torch.stack(visual_exp_trans)
#         tempD = F.l1_loss(att_map, visual_exp_trans)
#         trans_att_loss = trans_att_loss + tempD
    
#     elif trans_type == "HAICS":
#         temp_att_loss = BCE_criterion(att_map, visual_exp_trans * (visual_exp_trans > 0).float())
#         mask = (visual_exp_trans != 0).float()
#         trans_att_loss = torch.mean(temp_att_loss * mask)
#     elif trans_type == "GRADIA":
#         temp_att_loss = l1_criterion(att_map, visual_exp_trans * (visual_exp_trans > 0).float())
#         trans_att_loss = torch.mean(temp_att_loss)
#     return trans_att_loss

BCE_criterion = nn.BCELoss(reduction='none')
l1_criterion = nn.L1Loss(reduction="none")

def BF_solver(X, Y):
    epsilon = 1e-4

    with torch.no_grad():
        # Flatten the tensors along the last three dimensions (height, width, channels)
        x = X.view(X.size(0), -1)
        y = Y.view(Y.size(0), -1)
        
        g_idx = (y < 0)
        le_idx = (y > 0)
        
        # Initialize a and a_ct
        a = torch.zeros(X.size(0), device=X.device)
        a_ct = torch.zeros(X.size(0), device=X.device)

        # Compute the counts of g_idx and le_idx
        len_g = g_idx.sum(dim=1, keepdim=True).float()
        len_le = le_idx.sum(dim=1, keepdim=True).float()

        # Avoid division by zero
        len_g[len_g == 0] = 1.0
        len_le[len_le == 0] = 1.0

        # Compute v and v_ct for g_idx
        v_g = x + epsilon
        v_ct_g = torch.zeros_like(v_g)
        v_ct_g[g_idx] = (v_g.unsqueeze(2) > x.unsqueeze(1)).float().sum(dim=2) / len_g
        v_ct_g[le_idx] = (v_g.unsqueeze(2) <= x.unsqueeze(1)).float().sum(dim=2) / len_le
        
        max_v_ct_g, idx_g = v_ct_g.max(dim=1)
        a[max_v_ct_g > a_ct] = v_g.gather(1, idx_g.unsqueeze(1)).squeeze()[max_v_ct_g > a_ct]
        a_ct[max_v_ct_g > a_ct] = max_v_ct_g[max_v_ct_g > a_ct]

        # Compute v and v_ct for le_idx
        v_le = x
        v_ct_le = torch.zeros_like(v_le)
        v_ct_le[g_idx] = (v_le.unsqueeze(2) > x.unsqueeze(1)).float().sum(dim=2) / len_g
        v_ct_le[le_idx] = (v_le.unsqueeze(2) <= x.unsqueeze(1)).float().sum(dim=2) / len_le
        
        max_v_ct_le, idx_le = v_ct_le.max(dim=1)
        a[max_v_ct_le > a_ct] = v_le.gather(1, idx_le.unsqueeze(1)).squeeze()[max_v_ct_le > a_ct]
        a_ct[max_v_ct_le > a_ct] = max_v_ct_le[max_v_ct_le > a_ct]

    return a.unsqueeze(1)

def cal_trans_att_loss(att_map, visual_exp_trans, trans_type=None):
    '''
    Define the loss function for the attention map transformation
    '''
    if not trans_type:
        return torch.tensor(0.0, device=att_map.device)

    if trans_type == "Gaussian":
        a = BF_solver(att_map, visual_exp_trans)
        temp1 = torch.tanh(5 * (att_map - a))
        temp_loss = l1_criterion(temp1, visual_exp_trans)
        temp_size = (visual_exp_trans != 0).float()
        eff_loss = torch.sum(temp_loss * temp_size, dim=[1, 2, 3]) / torch.sum(temp_size, dim=[1, 2, 3])
        trans_att_loss = torch.relu(torch.mean(eff_loss))

        tempD = l1_criterion(att_map, visual_exp_trans).mean(dim=[1, 2, 3])
        trans_att_loss = trans_att_loss + tempD.mean()

    elif trans_type == "HAICS":
        temp_att_loss = BCE_criterion(att_map, visual_exp_trans * (visual_exp_trans > 0).float())
        mask = (visual_exp_trans != 0).float()
        trans_att_loss = torch.mean(temp_att_loss * mask, ).mean()

    elif trans_type == "GRADIA":
        temp_att_loss = l1_criterion(att_map, visual_exp_trans * (visual_exp_trans > 0).float())
        trans_att_loss = torch.mean(temp_att_loss, ).mean()

    return trans_att_loss



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
