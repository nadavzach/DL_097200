# Import Libraries
import numpy as np
import os
import torch
import PIL
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import visualize
import resnet
import glob
from imgaug import augmenters as iaa
import imgaug as ia
import torch.optim.lr_scheduler

import matplotlib.pyplot as plt

# from temp import CustomImageDataset
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.cuda.empty_cache()

writer = SummaryWriter('/home/student/runs/HW1_maskNN')

# -------------  CONTROL -------------------#

test = True
batch_size = 128
validation_split = .2
shuffle_dataset = True
random_seed = 42
epoch_num = 10


# -------------  DATA -------------------#


class CustomImageDataset(Dataset):
    def __init__(self, img_fns, transform=None, target_transform=None):
        self.img_fns = img_fns
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_fns)

    def __getitem__(self, idx):
        img_path = self.img_fns[idx]
        image = read_image(img_path).to(torch.float32)
        label = img_path[-5]
        if self.transform is not None:
            image = self.transform(np.array(image))

        label = np.array(float(label)).reshape(1)
        label = torch.tensor(label)
        return image, label


test_transformations = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.ToTensor(),
])


test_set = datasets.ImageFolder("/home/student/Desktop/test", transform=test_transformations)

# train_img_dir = '/home/student/Desktop/train'
# train_img_fns = glob.glob(os.path.join(train_img_dir, '*.jpg'))
# train_set = CustomImageDataset(img_fns=train_img_fns, transform=transformations)
# test_img_dir = '/home/student/Desktop/test'
# test_img_fns = glob.glob(os.path.join(test_img_dir, '*.jpg'))
# test_set = CustomImageDataset(img_fns=test_img_fns, transform=transformations)



# Creating PT data samplers and loaders:

test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)


# -------------  NN model -------------------#

class Net(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(57600, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = resnet.resnet34(pretrained=False, num_classes=2, input_channels=3)

if torch.cuda.is_available():
    print("$$$$$$$$$$$$$$$    using GPU    $$$$$$$$$$$$$$ ")
# Find the device available to use using torch library
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
# net = Net()


PATH = '/home/student/Desktop/HW1_net.pth'

net.load_state_dict(torch.load(PATH))
# ------------- TEST ------------- #
if test:
    net.eval()
    correct = 0
    total = 0
    tp = 0
    fp = 0
    fn = 0
    with torch.no_grad():
        for data in test_loader:
            # images, labels = data
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_vec = (predicted == labels)
            wrong_vec = (predicted != labels)
            correct += correct_vec.sum().item()
            tp += (correct_vec * labels).sum().item()
            fp += (wrong_vec * labels).sum().item()
            fn += (wrong_vec * (labels == 0)).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct // total} %')
    print(f'F1 score of the network on the test images: {2 * tp / (2 * tp + fp + fn)} ')
