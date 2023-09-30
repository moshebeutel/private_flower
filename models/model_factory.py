import torch.nn
from models.resnet_cifar import resnet20
from models.sixty_min_blitz_cnn import Net

models_hub: dict[str, torch.nn.Module] = {
    'blitz': Net(),
    'resnet20': resnet20()
}


def get_model_hub_names() -> list[str]:
    return list(models_hub.keys())


def get_model(model_name: str = 'blitz') -> torch.nn.Module:
    assert model_name in models_hub, f'Expected one of {models_hub.keys()}.\n Got {model_name}'
    return models_hub[model_name]
