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
# from pytorchtools import EarlyStopping

import matplotlib.pyplot as plt

# from temp import CustomImageDataset
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.cuda.empty_cache()

writer = SummaryWriter('./runs/HW1_maskNN')

# -------------  CONTROL -------------------#

test = True

validation_split = .2
shuffle_dataset = True
random_seed = 42

batch_size = 64
epoch_num = 16

initial_learning_rate = 0.2
exp_lr_schd_gamma = 0.9
# early_stopping_patience = 2

weights_save_name = "weigths_1.pth"
weights_save_location = '/home/student/Desktop/'
weights_save_path = os.path.join(weights_save_location, weights_save_name)

# -------------  PLOTTING -------------------#

y_loss = {}  # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []
x_epoch = []

fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")

running_loss = 0.0
running_corrects = 0.0


def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig('/home/student/Desktop/train.jpg')


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
        np.moveaxis(np.array(image), 0, -1)
        label = img_path[-5]
        if self.transform is not None:
            image = self.transform(image)

        label = np.array(float(label)).reshape(1)
        label = torch.tensor(label)
        return image, label



transformations = transforms.Compose([
#    transforms.ToPILImage(),
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.ColorJitter(hue=.05, saturation=.05),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])

test_transformations = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
])

train_set = datasets.ImageFolder("/home/student/Desktop/train", transform=transformations)
test_set = datasets.ImageFolder("/home/student/Desktop/test", transform=test_transformations)

#train_img_dir = '/home/student/Desktop/train'
#train_img_fns = glob.glob(os.path.join(train_img_dir, '*.jpg'))
#train_set = CustomImageDataset(img_fns=train_img_fns, transform=transformations)
#test_img_dir = '/home/student/Desktop/test'
#test_img_fns = glob.glob(os.path.join(test_img_dir, '*.jpg'))
#test_set = CustomImageDataset(img_fns=test_img_fns, transform=test_transformations)


# Creating data indices for training and validation splits:
dataset_size = len(train_set)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           sampler=train_sampler,
                                           pin_memory=True, num_workers=4)
validation_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                sampler=valid_sampler, pin_memory=True)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)


# -------------  NN model -------------------#


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=0, stride=4),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(3, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(3, stride=1))

        self.layer3 = nn.Sequential(
            nn.Conv2d(512, 384, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(384),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(3, stride=2))

        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01))

        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01))

        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(3, stride=2))

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 2)
        # self.fc3 = nn.Linear(256, 128)
        # self.fc4 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(p=0.5)
        self.leakyReLU = nn.LeakyReLU(0.01)
        # self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.leakyReLU(self.fc1(self.dropout(x)))
        x = self.fc2(x)
        # x = self.leakyReLU(self.fc2(self.dropout(x)))
        # x = self.leakyReLU(self.fc3(self.dropout(x)))
        # x = self.fc4(x)
        return x


net = resnet.resnet34(pretrained=False, num_classes=2, input_channels=3)
# net = Net1()
if torch.cuda.is_available():
    print("$$$$$$$$$$$$$$$    using GPU    $$$$$$$$$$$$$$ ")
# Find the device available to use using torch library
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
# aa

# -------------  optimizer -------------------#


optimizer = torch.optim.SGD(net.parameters(), lr=initial_learning_rate, momentum=0.9)
scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp_lr_schd_gamma)
scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, verbose=True)
criterion = nn.CrossEntropyLoss().cuda(device)

# early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)

# -------------  TRAINING -------------------#
print("--------------------- starting training ---------------------")

# Move model to the device specified above
net.train()

train_losses = []
train_corrects = []
valid_losses = []
valid_corrects = []
for epoch in range(epoch_num):  # loop over the dataset multiple times

    epoch_loss = 0.0
    epoch_acc = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        # inputs, labels = data
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # statistics
        train_losses.append(loss.item())
        train_corrects.append(float(torch.sum(predicted == labels)))

    epoch_train_loss = np.average(train_losses)
    epoch_train_acc = np.average(train_corrects)

    print(f'[{epoch + 1}, training loss: {epoch_train_loss:.3f}')

    with torch.no_grad():
        for i, data in enumerate(validation_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # inputs, labels = data
            inputs, labels = data[0].to(device), data[1].to(device)

            # forward + backward + optimize
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            # statistics
            valid_losses.append(loss.item())
            valid_corrects.append(float(torch.sum(predicted == labels)))

    epoch_valid_loss = np.average(valid_losses)
    epoch_valid_acc = np.average(valid_corrects)

    print(f'[{epoch + 1}, valid loss: {epoch_valid_loss:.3f}')
    y_loss['train'].append(np.reshape(epoch_train_loss, [1, ]))
    y_err['train'].append(1.0 - epoch_train_acc)

    y_loss['val'].append(np.reshape(epoch_valid_loss, [1, ]))
    y_err['val'].append(1.0 - epoch_valid_acc)
    draw_curve(epoch)

    # cleaning losses arrs
    train_losses = []
    train_corrects = []
    valid_losses = []
    valid_corrects = []

    # learning rate update
    scheduler1.step()
    scheduler2.step(epoch_valid_loss)

    # early_stopping(epoch_valid_loss, net)

    # if early_stopping.early_stop:
    #    print("Early stopping")
    #    break

print('Finished Training')

torch.save(net.state_dict(), weights_save_path)

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
