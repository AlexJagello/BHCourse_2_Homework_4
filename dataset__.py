
import os
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image



    
class ImageDataset(Dataset):
    def __init__(self, root, transform, iscolored):
        self.transform = transform
        self.root = root
        self.files = []
        self.iscolored = iscolored
        for file in os.listdir(root):
            if file.endswith(".jpg"):
                self.files.append(os.path.join(root, file))
        

    def __getitem__(self, index):
        img =[]
        if self.iscolored == False:
            img = Image.open(self.files[index]).convert('L')
        else: img = Image.open(self.files[index])
        
        if self.transform != None: 
            x = self.transform(img)
            return x
        return img

    def __len__(self):
        return len(self.files)
    



