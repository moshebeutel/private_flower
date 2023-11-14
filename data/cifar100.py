from functools import partial
from torchvision.datasets import CIFAR100
from data.dataset_wrapper import DatasetWrapper


def get_dataset_wrapper():
    return Cifar100Wrapper


def get_ood_dataset_wrapper():
    return Cifar100AugWrapper


class Cifar100Wrapper(DatasetWrapper):
    def __init__(self, root: str):
        dataset_ctor = partial(CIFAR100, root)
        super().__init__(dataset_ctor, ood=False)

    @property
    def classes(self) -> list[str]:
        return self._data_set.classes


class Cifar100AugWrapper(DatasetWrapper):
    def __init__(self, root: str):
        dataset_ctor = partial(CIFAR100, root)
        super().__init__(dataset_ctor, ood=True)

    @property
    def classes(self) -> list[str]:
        return self._data_set.classes
