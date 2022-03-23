import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from dataloader import load_cifar
from utils import accuracy, line_plot

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample

        ##TODO initialize weights: https://github.com/arjun-majumdar/CNN_Classifications/blob/master/ResNet-18_CIFAR10-PyTorch.ipynb

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.relu(self.bn1(x))

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)
        
        x += residual
        out = self.relu(x)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()

        self.input_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 , num_classes)

        ##TODO where is downsample used? and how to custom make layer?

    def _make_layer(self, block, out_channels, num_layers, stride):
        downsample = None

        if stride != 1 or self.input_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.input_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(block(in_channels=self.input_channels, out_channels=out_channels, downsample=downsample, stride=stride))
        self.input_channels = out_channels
        for _ in range(1, num_layers):
            layers.append(block(self.input_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1) # Here, torch.flatten(x, 1) can be used also, a nn.Flatten()
                                      # can be defiened in init after adaptive pool. Experiment!!!
        out = self.fc(x)

        return out

if __name__ == "__main__":

    print(f"Using {device} as the accelerator")
    batch_size = 1
    img_per_row = 8
    train_dataloader, valid_dataloader, test_dataloader = load_cifar(batch_size=batch_size, root='./data/')

    checkpoint = torch.load('./models/checkpoint_resnet_pre_ciphar.pth.tar')
    print("Found Checkpoint :)")
    classifier = checkpoint["classifier"]
    classifier.to(device)

    for i, (image, label) in enumerate(test_dataloader):
        image = image.to(device)
        for name, child in classifier.named_children():
            if name == "avgpool":
                break
            image = child(image)
            # if isinstance(child, nn.ReLU) or isinstance(child, nn.Sequential):
            image = image.to("cpu")
            n_features = image.shape[1]
            size = image.shape[-1]
            n_cols = n_features // img_per_row
            display_grid = np.zeros((size * n_cols, img_per_row * size))

            for col in range(n_cols): # Tiles each filter into a big horizontal grid
                for row in range(img_per_row):
                    channel_image = image[0,
                                            col * img_per_row + row,
                                            :, :]
                    channel_image = channel_image.cpu().detach().numpy()
                    channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
                    channel_image /= channel_image.std()
                    channel_image *= 64
                    channel_image += 128
                    #channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                    display_grid[col * size : (col + 1) * size, # Displays the grid
                                row * size : (row + 1) * size] = channel_image
            scale = 1. / size
            plt.figure(figsize=(scale * display_grid.shape[1],
                                scale * display_grid.shape[0]))
            # plt.figure(figsize=(display_grid.shape[1]*10,
            #                     display_grid.shape[0]*10))
            # print(display_grid.shape[1]*10, display_grid.shape[0]*10)
            # plt.rcParams["figure.figsize"] = display_grid.shape[1]*10, display_grid.shape[0]*10

            new_data = np.zeros(np.array(display_grid.shape) * 15)

            for j in range(display_grid.shape[0]):
                for k in range(display_grid.shape[1]):
                    new_data[j * 15: (j+1) * 15, k * 15: (k+1) * 15] = display_grid[j, k]

            plt.title(name)
            plt.grid(False)
            plt.imsave(f"./plots/{name}_{n_features}_{size}.png", new_data, dpi=600)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')

            print(image.shape)
            print(name)
            print("###################################################")
            image = image.to("cuda")

        break