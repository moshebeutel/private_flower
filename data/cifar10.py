import random
from functools import partial
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
from data.dataset_wrapper import DatasetWrapper

#
# def load_data(root: str, batch_size=32, ood=False):
#     """
#     Load CIFAR-10 dataset.
#
#     Parameters:
#     root (str): Root directory where the dataset will be stored.
#     batch_size (int, optional): Batch size for data loaders. Defaults to 32.
#     ood (bool, optional): Flag indicating if the data is out-of-distribution (OOD). Defaults to False.
#
#     Returns:
#     Tuple[DataLoader, DataLoader]: DataLoader for the training set and DataLoader for the test set.
#     """
#     assert os.path.exists(root), f'{root} does not exist'
#
#     transform = transforms.Compose(
#         [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
#     ) if not ood else transforms.Compose(
#         [transforms.ToTensor(), transforms.ColorJitter(contrast=0.5, brightness=1.0),
#          transforms.Normalize((1.0, 0.25, 0.1), (0.5, 0.5, 0.5))]
#     )
#
#     print(f'load_data to {root}, ood? {ood}')
#     train_set = CIFAR10(root, train=True, download=True, transform=transform)
#     test_set = CIFAR10(root, train=False, download=True, transform=transform)
#     train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_set, batch_size=32)
#     return train_loader, test_loader

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def load_raw_data(root: str) -> tuple[Dataset, Dataset, Dataset, list[str]]:
    train_set = CIFAR10(root, train=True, download=True)

    train_set, validation_set = train_val_split(train_set)
    test_set = CIFAR10(root, train=False, download=True)

    return train_set, validation_set, test_set, CIFAR10_CLASSES


def train_val_split(train_set: Dataset, split_ratio: float = 0.8, shuffle: bool = True) -> tuple[Dataset, Dataset]:
    from torch.utils.data import DataLoader, Subset
    dataset_size = len(train_set)
    split_index = int(dataset_size * split_ratio)
    indices = list(range(dataset_size))
    if shuffle:
        random.shuffle(indices)
    # Create Subset objects for the training and validation sets
    train_subset = Subset(train_set, indices[:split_index])
    validation_subset = Subset(train_set, indices[split_index:])
    return train_subset, validation_subset


def get_dataset_wrapper(ood: bool):
    return Cifar10Wrapper if not ood else Cifar10AugWrapper


class Cifar10Wrapper(DatasetWrapper):
    def __init__(self, root: str):
        dataset_ctor = partial(CIFAR10, root)
        super().__init__(dataset_ctor, ood=False)

    @property
    def classes(self) -> list[str]:
        return CIFAR10_CLASSES


class Cifar10AugWrapper(DatasetWrapper):
    def __init__(self, root: str):
        dataset_ctor = partial(CIFAR10, root)
        super().__init__(dataset_ctor, ood=True)

    @property
    def classes(self) -> list[str]:
        return CIFAR10_CLASSES

#
