# Implementation of seam carving for CNN

import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from class_SeamCarver import SeamCarver  # Importing the SeamCarver class
import numpy as np



class SeamCarvingPooling(nn.Module): # SeamCarvingPooling class for PyTorch
    def __init__(self, num_seams=1, method='l1', orientation='vertical'):
        super().__init__()
        self.num_seams = num_seams
        self.method = method
        self.orientation = orientation

    def forward(self, x):
        """ Forward pass for SeamCarvingPooling """
        # x : [1, C, H, W]
        assert x.shape[0] == 1, "SeamCarvingPooling ne supporte que batch_size=1"
        x_np = x[0]  # [C, H, W]
        x_np = x_np.detach()    # Detach from the computation graph to avoid memory issues calculating gradients
        x_np = x_np.cpu()       # Move to CPU because numpy does not support GPU tensors
        x_np = x_np.numpy()     # Convert to numpy array

        out_channels = []  #  Liste des canaux traités
        for c in range(x_np.shape[0]):
            # Normalisation du canal sur [0, 255]
            img = (255 * (x_np[c] - x_np[c].min()) / (np.ptp(x_np[c]) + 1e-8)).astype(np.uint8)
            # Création image RGB factice
            img = np.stack([img] * 3, axis=-1)
            # Application du seam carving
            sc = SeamCarver(img)
            reduced = sc.seam_carve(self.num_seams, self.method, self.orientation)
            # Extraction d’un seul canal
            reduced_gray = reduced[:, :, 0]
            # Conversion en tenseur normalisé
            out_channels.append(torch.tensor(reduced_gray, dtype=torch.float32) / 255.0)

        #  Empilement → [C, H', W']
        out_tensor = torch.stack(out_channels, dim=0).to(x.device)
        return out_tensor.unsqueeze(0)  # → [1, C, H', W']

    def _tensor_to_numpy_img(self,tensor): 
        """ Convert PyTorch tensor to numpy image """
        if tensor.dim() == 4:       # We convert a batch of images in [B, C, H, W] format to a single image for seam carving
            tensor = tensor[0]      # [B, C, H, W] → [C, H, W]
        tensor = tensor.detach()    # Detach from the computation graph to avoid memory issues calculating gradients
        tensor = tensor.cpu()       # Move to CPU because numpy does not support GPU tensors
        tensor = tensor.numpy()     # Convert to numpy array
        # use mean to merge channels for OpenCV seam carving
        merged = np.mean(tensor, axis=0)  # [H, W]
        img = np.stack([merged] * 3, axis=-1)  # [H, W] → [H, W, 3]
        img = np.moveaxis(tensor, 0, -1)                # [C, H, W] → [H, W, C]
        img = (255 * (img - img.min()) / (np.ptp(img) + 1e-8)) # Normalize to [0, 255] range and avoid division by zero
        img = img.astype(np.uint8)                      # Convert to uint8 for image representation
        return img

    def _numpy_to_tensor(self, img): 
        """ Convert numpy image to PyTorch tensor """
        tensor = torch.tensor(img, dtype=torch.float32) / 255.0
        tensor = tensor.permute(2, 0, 1)  # [H, W, C] → [C, H, W]
        return tensor


class CNNWithSeamCarving(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)       # 3 → 16 canaux
        self.pool1 = SeamCarvingPooling(num_seams=16)                 # Réduction intelligente
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)      # 16 → 32 canaux
        self.pool2 = SeamCarvingPooling(num_seams=16)                   # Dimension fixe
        self.fc = nn.Linear(32 * 16 * 16, 2)                           # Seulement 2 classes

    def forward(self, x):
        x = F.relu(self.conv1(x))         # conv1 + ReLU
        x = self.pool1(x)                 # Seam carving
        x = F.relu(self.conv2(x))         # conv2 + ReLU
        x = self.pool2(x)                 # pooling final
        x = x.view(x.size(0), -1)         # flatten
        return self.fc(x)                 # prédiction binaire


    
class CNNWithMaxPooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.AdaptiveAvgPool2d((16, 16))
        self.fc = nn.Linear(32 * 16 * 16, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)



