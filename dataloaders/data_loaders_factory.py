import os

from torch.utils.data import DataLoader

from dataloaders.cifar10_loader import load_data


def get_data_loaders(data_path: str,
                     batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Get data loaders for CIFAR-10 dataset.

    Parameters:
    data_path (str): Root directory where the dataset will be loaded.
    batch_size (int): Batch size for the data loaders.

    Returns:
    Tuple[DataLoader, DataLoader, DataLoader, DataLoader, DataLoader, DataLoader]: Data loaders for training set,
                                                           out-of-distribution training set, validation set,
                                                           out-of-distribution validation set,
                                                           test set, and out-of-distribution test set.
    """
    assert os.path.exists(data_path), f'Path given to data does not exist. Got {data_path}'

    # Load data for standard training and testing
    train_loader, val_loader, test_loader = load_data(root=data_path, batch_size=batch_size, ood=False)

    # Load data for out-of-distribution (OOD) training and testing
    train_loader_ood, val_loader_ood, test_loader_ood = load_data(root=data_path, batch_size=batch_size, ood=True)

    return train_loader, train_loader_ood, val_loader, val_loader_ood, test_loader, test_loader_ood
