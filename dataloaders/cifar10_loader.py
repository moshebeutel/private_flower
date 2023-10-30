import os.path
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset


def load_data(root: str, batch_size: int = 32,
              validation_split: float = 0.1, ood: bool = False) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load CIFAR-10 dataset.

    Parameters:
    root (str): Root directory where the dataset will be stored.
    batch_size (int, optional): Batch size for data loaders. Defaults to 32.
    validation_split (float, optional): validation subset fraction in (0,1) interval out of train set. Defaults to 0.1.
    ood (bool, optional): Flag indicating if the data is out-of-distribution (OOD). Defaults to False.

    Returns:
    Tuple[DataLoader, DataLoader]: DataLoader for the training set and DataLoader for the test set.
    """
    assert os.path.exists(root), f'{root} does not exist'
    assert 0.0 < validation_split < 1.0, f'Expected validation_split in (0,1) interval. Got {validation_split}'

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    ) if not ood else transforms.Compose(
        [transforms.ToTensor(), transforms.ColorJitter(contrast=0.5, brightness=1.0),
         transforms.Normalize((1.0, 0.25, 0.1), (0.5, 0.5, 0.5))]
    )

    print(f'load_data to {root}, ood? {ood}')
    train_set = CIFAR10(root, train=True, download=True, transform=transform)
    validation_set_size = int(len(train_set) * validation_split)
    val_set = Subset(train_set, range(0, validation_set_size))
    test_set = CIFAR10(root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=32)
    return train_loader, val_loader, test_loader
