import os
import torch
from torch import nn
from torchvision.io import read_image
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class SymbolDataset(Dataset):
    def __init__(self, image_paths, classes, transform=None):
        self.image_paths = image_paths
        self.classes = classes
        self.transform = transform
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')
        
        parent_dir = os.path.dirname(img_path)
        label_name = os.path.basename(parent_dir)
        label = self.class_to_idx[label_name]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    