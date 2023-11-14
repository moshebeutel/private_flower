import os
from pathlib import Path

from torch.utils.data import DataLoader

from data import cifar10, cifar100

datasets_hub = {
    'CIFAR10': cifar10,
    'CIFAR100': cifar100
}


def get_datasets_hub_names() -> list[str]:
    return list(datasets_hub.keys())


def get_data_sets(data_path: str, dataset_name: str, ood: bool):
    dataset_module = datasets_hub[dataset_name]
    assert hasattr(dataset_module, 'get_dataset_wrapper'), (f'Expected a module that contains a DatasetWrapper. '
                                                            f'Got {dataset_module}')
    dataset_wrapper = dataset_module.get_dataset_wrapper(ood)(data_path)
    train_set, val_set, test_set = dataset_wrapper.get_dataset(splits='all')
    return train_set, val_set, test_set, dataset_wrapper.classes


# def get_data_sets(args, ood: bool = False):
#     root = Path(args.data_path)
#     assert root.is_dir(), f'Expected path to dataset root. Got non existent {root}'
#
#     dataset_name = args.dataset_name
#     assert dataset_name in datasets_hub, f'Expected one of {get_datasets_hub_names()}. Got {dataset_name}.'
#     train_set, val_set, test_set, classes = get_data_sets(data_path=root.as_posix(), dataset_name=dataset_name, ood=ood)
#
#     return train_set, val_set, test_set, classes


def get_data_loaders(data_path: str, dataset_name: str, batch_size: int):
    """
    Get data loaders for CIFAR-10 dataset.

    Parameters:
    data_path (str): Root directory where the dataset will be loaded.
    batch_size (int): Batch size for the data loaders.

    Returns:
    Tuple[DataLoader, DataLoader, DataLoader, DataLoader]: Data loaders for training set,
                                                           out-of-distribution training set,
                                                           test set, and out-of-distribution test set.
    """
    assert os.path.exists(data_path), f'Path given to data does not exist. Got {data_path}'

    # Load data for standard training and testing
    # train_loader, test_loader = load_data(root=data_path, batch_size=batch_size, ood=False)
    train_set, _, test_set, _ = get_data_sets(data_path=data_path, dataset_name=dataset_name, ood=False)

    # Load data for out-of-distribution (OOD) training and testing
    # train_loader_ood, test_loader_ood = load_data(root=data_path, batch_size=batch_size, ood=True)
    train_set_ood, _, test_set_ood, _ = get_data_sets(data_path=data_path, dataset_name=dataset_name, ood=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    train_loader_ood = DataLoader(train_set_ood, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    test_loader_ood = DataLoader(test_set_ood, batch_size=batch_size, shuffle=True)

    return train_loader, train_loader_ood, test_loader, test_loader_ood

# def get_raw_data(data_path: str):
#     train_set, test_set, classes = load_raw_data(data_path)
#     return train_set, test_set, classes
