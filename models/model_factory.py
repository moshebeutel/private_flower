import torch.nn
from models.resnet_cifar import *
from models.sixty_min_blitz_cnn import Net

models_hub: dict[str, torch.nn.Module] = {
    'blitz': Net(),
    'resnet20': resnet20(),
    'resnet32': resnet32(),
    'resnet44': resnet44(),
    'resnet56': resnet56(),
    'resnet110': resnet110(),
    'resnet1202': resnet1202(),

}


def get_model_hub_names() -> list[str]:
    return list(models_hub.keys())


def get_model(model_name: str = 'blitz') -> torch.nn.Module:
    assert model_name in models_hub, f'Expected one of {models_hub.keys()}.\n Got {model_name}'
    return models_hub[model_name]
