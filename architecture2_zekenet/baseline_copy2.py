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
# from torchsummary import summary

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
            nn.Conv2d(3, 11 , 3, stride=1, padding=1, bias=False), 
            nn.ReLU(inplace=True),
            nn.Conv2d(11, 22, 3,stride=2, padding=1, bias=False),
            nn.BatchNorm2d(22),
            nn.SELU( inplace=True), #nn.ReLU(inplace=True),
            nn.Conv2d(22, 33, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(33),
            nn.ReLU( inplace=True),
            nn.Conv2d(33, 44, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(44),
#             nn.ReLU( inplace=True)
            nn.SELU( inplace=True),
            nn.MaxPool2d(5, stride=5), #maxpool makes ofm 6x6
        )
        self.fc = nn.Sequential(
            nn.Linear(44 * 6 * 6, 503),
            nn.ReLU( inplace=True),
            nn.Linear(503, 201),
            
            #COMMENT THIS DURING TRAINING
            #UNCOMMENT DURING TESTING
            nn.Softmax()
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
        x=self.main(input)
        x=x.view(-1, self.num_flat_features(x))
        return self.fc(x)

    
        
