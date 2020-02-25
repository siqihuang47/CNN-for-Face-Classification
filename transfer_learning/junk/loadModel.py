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
#from torchsummary import summary
from baseline_cnn import *


def load_model(model_name):
    
    device = torch.device('cuda:0')
    model = torch.load(model_name)

    model.to(device)
    #enter eval mode
    #model.eval()
    
    return model

#     transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()])
#     dataset = loader('test.csv','/datasets/cs154-fa19-public/',transform=transform)
#     test_loader = torch.utils.data.DataLoader(dataset)

# # print(len(test_loader))
# classes_right = [0] * 201
# # classes_right[0] = 1
# # class_images = []
# # for images, labels in dataset:
# #     class_labels.append(labels)
# # #     class_images.append(images)
    
# # # print(np.amax(class_labels), np.amin(class_labels))
# # # c = 0

# classes_count = [0] * 201
# # print(classes_count, len(classes_count))
# # print(classes_count)

# pred_classes = []
# accu = 0
# with torch.no_grad():
#     for i, (images, labels) in enumerate(test_loader, 0):
#         y = model(images)
#         t = labels.item()
#         classes_count[t] += 1
#         y_i = torch.argmax(y).item()
#         pred_classes.append(y_i)
# #         print(i)
# #         print(images)
# #         print(labels)
# #         print(labels.item())
#         if t == y_i:
#             accu += 1
#             classes_right[t] += 1
            
# # print(pred_classes)
# print(np.amax(pred_classes), np.amin(pred_classes))
# # # print(y)
# # # print(torch.argmax(y).item())
# print("accuracy: ", (accu/len(dataset)))
# print(accu)
# classes_accu = np.array(classes_right[1:]) / np.array(classes_count[1:])
# #b = best, w = worst
# b = np.argmax(classes_accu)
# w = np.argmin(classes_accu)

# print("best class performace with (accuracy): ", b+1, "(" + str(classes_accu[b]) + ")", np.amax(classes_accu), " --- " + str(classes_right[b+1]) + "/" + str(classes_count[b+1]) + " right")

# print("worst class performace with (accuracy): ", w+1, "(" + str(classes_accu[w]) + ")", np.amin(classes_accu), " --- " + str(classes_right[w+1]) + "/" + str(classes_count[w+1]) + " right")