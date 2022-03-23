import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, random_split, Subset

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_mnist(batch_size: int=64, root: str='./data/'):
    """
    Load MNIST data
    """
    t = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Lambda(lambda x: torch.flatten(x))]
                       )
    train = datasets.MNIST(root=root, train=True, download=True, transform=t)
    train_data, valid_data = random_split(train, [50000, 10000])
    test_data  = datasets.MNIST(root=root, train=False, download=True, transform=t)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, drop_last=True, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, drop_last=True, shuffle=True)
    test_dataloader  = DataLoader(test_data, batch_size=batch_size, drop_last=True, shuffle=True)

    return train_dataloader, valid_dataloader, test_dataloader

def load_cifar(batch_size: int=64, root: str="./data/"):
    """
    Load CIFAR-10 data
    """
    transform_train = transforms.Compose([
        #   transforms.RandomCrop(32, padding = 4),
        #   transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
    train = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    train_data, valid_data = random_split(train, [45000, 5000])
    test_data  = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, drop_last=True, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, drop_last=True, shuffle=True)
    test_dataloader  = DataLoader(test_data, batch_size=batch_size, drop_last=True, shuffle=True)

    return train_dataloader, valid_dataloader, test_dataloader

def load_tiny_cifar(batch_size: int=64, root: str="./data/"):
    """
    Load CIFAR-10 data
    """
    transform_train = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.RandomCrop(32, padding = 4),
        transforms.RandomAffine(degrees=(30, 45)),
        transforms.RandomAdjustSharpness(sharpness_factor=1.2),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        # transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.05),
        # transforms.RandomGrayscale(p=0.25),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
    train = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)

    count = {i: 0 for i in range(10)}
    keep_list = []

    def full(ct):
        for val in ct.values():
            if val < 500:
                return False
        
        return True

    for idx in range(len(train)):
        label = train[idx][1]
        if count[label] < 500:
            count[label] += 1
            keep_list.append(idx)
        if full(count):
            break

    train = Subset(train, keep_list)
    train_data, valid_data = random_split(train, [4500, 500])
    test_data  = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, drop_last=True, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, drop_last=True, shuffle=True)
    test_dataloader  = DataLoader(test_data, batch_size=batch_size, drop_last=True, shuffle=True)

    return train_dataloader, valid_dataloader, test_dataloader

if __name__ == "__main__":
    train_dataloader, valid_dataloader, test_dataloader = load_tiny_cifar(root='./data/')
    print(len(train_dataloader) * 64, len(test_dataloader) * 64)
    # for x, y in train_dataloader:
    #     print(x[0])
    #     print(y[0])
    #     break