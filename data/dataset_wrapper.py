from abc import ABC
import random
import torchvision.transforms.v2 as v2
from torch.utils.data import Dataset, Subset

transforms_hub = {
    'standard': v2.Compose([v2.ToTensor(),
                            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'ood': v2.Compose(
        [v2.ToTensor(), v2.ColorJitter(contrast=0.5, brightness=1.0),
         v2.Normalize((1.0, 0.25, 0.1), (0.5, 0.5, 0.5))])
}


def train_val_split(train_set: Dataset, split_ratio: float = 0.8, shuffle: bool = True) -> tuple[Dataset, Dataset]:
    dataset_size = len(train_set)
    split_index = int(dataset_size * split_ratio)
    indices = list(range(dataset_size))
    if shuffle:
        random.shuffle(indices)
    # Create Subset objects for the training and validation sets
    train_subset = Subset(train_set, indices[:split_index])
    validation_subset = Subset(train_set, indices[split_index:])
    return train_subset, validation_subset


class DatasetWrapper(ABC):
    def __init__(self, datset_ctor, ood: bool = False):
        self._data_set = datset_ctor(train=True, download=True, transform=transforms_hub['ood' if ood else 'standard'])

        train_set, validation_set = train_val_split(self._data_set)
        test_set = datset_ctor(train=False, download=True)
        self._train_set = train_set
        self._validation_set = validation_set
        self._test_set = test_set

        self._split_to_ds = {'train': self._train_set, 'validation': self._validation_set,
                             'test': self._test_set, 'val': self._validation_set,
                             'all': (self._train_set, self._validation_set, self._test_set)}

    def get_dataset(self, splits: str | list[str]):
        if isinstance(splits, str):
            splits = [splits]
        if 'all' in splits:
            return self._split_to_ds['all']
        datasets = []
        for split in splits:
            assert split in self._split_to_ds, f'Expected one of {self._split_to_ds.keys()}. Got {split}'
            datasets.append(self._split_to_ds[split])
        return tuple(set(datasets))

    @property
    def train_set(self):
        return self._train_set

    @property
    def validation_set(self):
        return self._validation_set

    @property
    def test_set(self):
        return self._test_set

    @property
    def classes(self) -> list[str]:
        raise NotImplementedError
