import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

from dataset import PascalVOCDataset, PascalVOCDataset2

data_folder = "./"

def load_mnist(batch_size: int=64, root: str='./data/'):
    """
    Load MNIST data
    """
    t = transforms.Compose([
                       transforms.ToTensor(),
                       ]
                       )
    train_data = datasets.MNIST(root=root, train=True, download=True, transform=t)
    test_data  = datasets.MNIST(root=root, train=False, download=True, transform=t)
    
    train_dataloader = DataLoader(train_data, batch_size=batch_size, drop_last=True, shuffle=True)
    test_dataloader  = DataLoader(test_data, batch_size=batch_size, drop_last=True, shuffle=True)

    return train_dataloader, test_dataloader


def load_cifar(batch_size: int=64, root: str="./data/", return_set=False):
    """
    Load CIFAR-10 data
    """
    transform_train = transforms.Compose([
        # transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop((32,32),scale=(0.8,1.0),ratio=(0.9,1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
        # transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
    train = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    train_data, valid_data = random_split(train, [45000, 5000])
    test_data  = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, drop_last=True, num_workers=4)
    test_dataloader  = DataLoader(test_data, batch_size=batch_size, drop_last=True, num_workers=4)

    if return_set == True:
        return valid_data

    return train_dataloader, valid_dataloader, test_dataloader


def load_pascal(batch_size=64):
    train_dataset = PascalVOCDataset(data_folder,
                                     split='train',
                                     keep_difficult=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=4,
                                               pin_memory=True)
    test_dataset = PascalVOCDataset2(data_folder,
                                     split='test',
                                     keep_difficult=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              collate_fn=test_dataset.collate_fn, num_workers=4, 
                                              pin_memory=True)

    return train_loader, test_loader

if __name__ == "__main__":
    load_pascal(1)