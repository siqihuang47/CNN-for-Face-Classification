import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim
import pandas as pd
import torchvision
from torchvision import transforms, utils
import os

class loader(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.transform = transform
        self.frame = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,self.frame.iloc[idx, 0])
        img = Image.open(img_name)
        label = self.frame.iloc[idx, 1]
        
        if self.transform:
            img = self.transform(img)

        return img, label
    
    
class Nnet(nn.Module):
    def __init__(self):
        super(Nnet, self).__init__()
        self.main = nn.Sequential(
            
            nn.Conv2d(3, 15, 5, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(15, 35, 5, padding=2),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(35, 50, 3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(50, 50, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(2450, 1000),
            nn.ReLU( inplace=True),
            nn.Linear(1000, 201),
        )

    def num_flat_features(self, inputs):
        
        # Get the dimensions of the layers excluding the inputs
        size = inputs.size()[1:]
        # Track the number of features
        num_features = 1
        
        for s in size:
            num_features *= s
        
        return num_features

    def forward(self, input):
        x = self.main(input)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc(x)
        m = nn.Sigmoid()
        return m(x)

        
