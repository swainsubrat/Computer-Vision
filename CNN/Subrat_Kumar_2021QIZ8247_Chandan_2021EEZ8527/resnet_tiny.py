import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

import numpy as np
import torch.nn as nn

from dataloader import load_tiny_cifar
from utils import accuracy, line_plot

def save_checkpoint(epoch, classifier, optimizer, path='./models/checkpoint_resnet_tiny_ciphar.pth.tar'):
    state = {'epoch': epoch,
             'classifier': classifier,
             'optimizer': optimizer}

    filename = path
    torch.save(state, filename)

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

    # net = ResNet(block=ResNetBlock, layers=[2, 2, 2, 2], num_classes=10)
    batch_size = 128
    epochs     = 100
    patience   = 100
    init_lr    = 0.001
    grad_clip  = 0.1
    weight_decay = 1e-4
    min_valid_loss = np.inf
    
    print(f"Using {device} as the accelerator")
    train_dataloader, valid_dataloader, test_dataloader = load_tiny_cifar(batch_size=batch_size, root='./data/')
    try:
        # try loading checkpoint
        checkpoint = torch.load('./models/checkpoint_resnet_tiny_ciphar.pth.tar')
        print("Found Checkpoint :)")
        classifier = checkpoint["classifier"]
        classifier.to(device)

    except:
        # train the model from scratch
        print("Couldn't find checkpoint :(")

        classifier = ResNet(block=ResNetBlock, layers=[2, 2, 2, 2], num_classes=10)

        def append_dropout(model, rate=0.2):
            for name, module in model.named_children():
                if len(list(module.children())) > 0:
                    append_dropout(module)
                if isinstance(module, nn.ReLU):
                    new = nn.Sequential(module, nn.Dropout2d(p=rate))
                    setattr(model, name, new)
        
        append_dropout(classifier)

        classifier.to(device)
        criterion = nn.CrossEntropyLoss().to(device)

        optimizer = torch.optim.Adam(classifier.parameters(), lr=init_lr, weight_decay=weight_decay)

        # optimizer = torch.optim.SGD(classifier.parameters(), lr=init_lr, momentum=0.9, weight_decay=5e-4)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        train_los = []
        valid_los = []
        train_acc  = []
        valid_acc  = []
        epoch_since_last_improve = 0
        for epoch in range(epochs):
            train_loss = 0.0
            y_hat, y = [], []
            classifier.train()
            for i, (image, label) in enumerate(train_dataloader):
                image = image.to(device)
                label = label.to(device)
                out = classifier(image)
                
                loss = criterion(out, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                if i % 100 == 0 and i != 0:
                    print(f"Epoch: [{epoch+1}][{i}/{len(train_dataloader)}] Loss: {loss.item(): .4f}")
                
                prob, idxs = torch.max(out, dim=1)
                y.extend(label.tolist())
                y_hat.extend(idxs.tolist())
            
            train_acc.append(accuracy(y, y_hat))

            valid_loss = 0.0
            classifier.eval()
            y_hat, y = [], []

            for i, (image, label) in enumerate(valid_dataloader):
                image = image.to(device)
                label = label.to(device)
                out = classifier(image)
                loss = criterion(out, label)
                valid_loss = loss.item() * batch_size

                prob, idxs = torch.max(out, dim=1)
                y.extend(label.tolist())
                y_hat.extend(idxs.tolist())
            
            valid_acc.append(accuracy(y, y_hat))
            
            # scheduler.step()
                
            print(
                f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_dataloader): .4f} \
                \t\t Validation Loss: {valid_loss / len(valid_dataloader): .4f}'
            )

            epoch_since_last_improve += 1
            if valid_loss < min_valid_loss:
                print(f'Validation Loss Decreased({min_valid_loss:.4f}--->{valid_loss:.4f}) \t Saving The Model')
                min_valid_loss = valid_loss
                epoch_since_last_improve = 0
                save_checkpoint(epoch, classifier, optimizer)
            
            train_los.append(train_loss / len(train_dataloader))
            valid_los.append(valid_loss / len(valid_dataloader))
            
            print(f"Epochs since last improvement: {epoch_since_last_improve}")
            if epoch_since_last_improve > patience:
                print("Early stopping! Breaking out of loop....")
                break
    
        line_plot(train_acc, valid_acc, "Accuracy vs Epochs", "Epochs", "Accuracy", "./plots/mnist_reg_acc.png")
        line_plot(train_los, valid_los, "Loss vs Epochs", "Epochs", "Loss", "./plots/mnist_reg_loss.png")

    y_hat, y = [], []
    for i, (image, label) in enumerate(test_dataloader):
        image = image.to(device)
        out = classifier(image)
        prob, idxs = torch.max(out, dim=1)

        y.extend(label.tolist())
        y_hat.extend(idxs.tolist())

    print(accuracy(y, y_hat))

    # from torchvision import models
    # print(models.resnet18())

    # print(net == models.resnet18())