from sklearn.utils import shuffle
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, random_split

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


if __name__ == "__main__":
    train_dataloader, valid_dataloader, test_dataloader = load_mnist(root='./data/')
    print(len(train_dataloader) * 64, len(test_dataloader) * 64)
    for x, y in train_dataloader:
        # print(x[0])
        # print(y[0])
        break