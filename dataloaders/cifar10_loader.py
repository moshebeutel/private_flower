import os.path

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


def load_data(root: str):
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    assert os.path.exists(root), f'{root} does not exist'
    trainset = CIFAR10(root, train=True, download=True, transform=transform)
    testset = CIFAR10(root, train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32)
    return trainloader, testloader
