import torch
from torchvision import datasets, transforms

def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

def get_train_set(download_path):
    return datasets.MNIST(download_path, train=True, download=True, transform=get_transform())

def get_test_set(download_path):
    return datasets.MNIST(download_path, train=False, download=True, transform=get_transform())

def train_loader(download_path, batch_size_train):
    train_set = get_train_set(download_path)
    return torch.utils.data.DataLoader(train_set, batch_size=batch_size_train, shuffle=True)

def test_loader(download_path, batch_size_test):
    test_set = get_test_set(download_path)
    return torch.utils.data.DataLoader(test_set, batch_size=batch_size_test, shuffle=True)
