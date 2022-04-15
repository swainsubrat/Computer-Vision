"""
A module to benchmark transformer against other CNN models
AlexNet, VGG-Net, Inception-Net, ResNet
"""
import torch
import numpy as np

from torchvision import models
from torchsummary import summary

from dataloader import load_cifar

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

alexnet   = models.alexnet(pretrained=True)
vggnet    = models.vgg16(pretrained=True)
resnet18  = models.resnet18(pretrained=True)
inception = models.inception_v3(pretrained=True)

pretrained_models = [alexnet, vggnet, resnet18, inception]
model_names       = ["AlexNet", "VGGNet", "ResNet", "InceptionNet"]

train_loader, val_loader, test_loader = load_cifar(batch_size=64, root='./data/')

def accuracy(Y, predY) -> float:
    """
    Get accuracy
    """
    Y = np.array(Y)
    predY = np.array(predY)
    accuracy = (Y == predY).sum()/ float(len(Y))
    accuracy = np.round(accuracy * 100, 2)

    return accuracy

def compare(model: str, model_name: str):
    classifier = model.to(device)
    classifier.eval()

    y_hat, y = [], []
    for i, (image, label) in enumerate(test_loader):
        image = image.to(device)
        out = classifier(image)
        prob, idxs = torch.max(out, dim=1)

        y.extend(label.tolist())
        y_hat.extend(idxs.tolist())

    print(f"Test accuracy for model {model_name}: {accuracy(y, y_hat)}")

    # y_hat, y = [], []
    # for i, (image, label) in enumerate(val_loader):
    #     image = image.to(device)
    #     out = classifier(image)
    #     prob, idxs = torch.max(out, dim=1)

    #     y.extend(label.tolist())
    #     y_hat.extend(idxs.tolist())

    # print(f"Validation accuracy for model {model_name}: {accuracy(y, y_hat)}")

for model, model_name in zip(pretrained_models, model_names):
    compare(model, model_name)
