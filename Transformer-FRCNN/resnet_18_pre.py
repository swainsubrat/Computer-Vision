import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

import numpy as np
import torch.nn as nn

from dataloader import load_cifar
from torchvision import models

def accuracy(Y, predY) -> float:
    """
    Get accuracy
    """
    Y = np.array(Y)
    predY = np.array(predY)
    accuracy = (Y == predY).sum()/ float(len(Y))
    accuracy = np.round(accuracy * 100, 2)

    return accuracy

def save_checkpoint(epoch, classifier, optimizer, path='./saved_models/checkpoint_resnet50_pre_ciphar.pth.tar'):
    state = {'epoch': epoch,
             'classifier': classifier,
             'optimizer': optimizer}

    filename = path
    torch.save(state, filename)

net = models.resnet50(pretrained=True)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 10)

for p in net.parameters():
    p.requires_grad = False

for c in list(net.children())[5:]:
    for p in c.parameters():
        p.requires_grad = True

if __name__ == "__main__":

    # net = ResNet(block=ResNetBlock, layers=[2, 2, 2, 2], num_classes=10)
    batch_size = 128
    epochs     = 30
    patience   = 5
    init_lr    = 0.01
    grad_clip  = 0.1
    weight_decay = 1e-4
    min_valid_loss = np.inf
    
    print(f"Using {device} as the accelerator")
    train_dataloader, valid_dataloader, test_dataloader = load_cifar(batch_size=batch_size, root='./data/')
    try:
        # try loading checkpoint
        checkpoint = torch.load('./saved_models/checkpoint_resnet50_pre_ciphar.pth.tar')
        print("Found Checkpoint :)")
        classifier = checkpoint["classifier"]
        classifier.to(device)

    except:
        # train the model from scratch
        print("Couldn't find checkpoint :(")

        # classifier = ResNet(block=ResNetBlock, layers=[2, 2, 2, 2], num_classes=10)
        classifier = net
        classifier.to(device)
        criterion = nn.CrossEntropyLoss().to(device)

        # optimizer = torch.optim.Adam(classifier.parameters(), lr=init_lr, weight_decay=weight_decay)
        # sched	  = torch.optim.lr_scheduler.OneCycleLR(optimizer, init_lr, epochs=epochs, steps_per_epoch=len(train_dataloader))
        #optimizer = torch.optim.SGD(classifier.parameters(), lr = 0.01, momentum = 0.9, weight_decay = 5e-4)

        optimizer = torch.optim.SGD(classifier.parameters(), lr=init_lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

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
            
            scheduler.step()
                
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
    
        # line_plot(train_acc, valid_acc, "Accuracy vs Epochs", "Epochs", "Accuracy", "./plots/mnist_reg_acc.png")
        # line_plot(train_los, valid_los, "Loss vs Epochs", "Epochs", "Loss", "./plots/mnist_reg_loss.png")

        print(f"Train Accuracy: {train_acc[-1]}")

    y_hat, y = [], []
    for i, (image, label) in enumerate(test_dataloader):
        image = image.to(device)
        out = classifier(image)
        prob, idxs = torch.max(out, dim=1)

        y.extend(label.tolist())
        y_hat.extend(idxs.tolist())

    print(f"Test Accuracy: {accuracy(y, y_hat)}")
