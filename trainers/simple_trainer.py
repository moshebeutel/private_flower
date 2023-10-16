from typing import List
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(net: torch.nn.Module, train_loader: DataLoader, epochs: int = 1):
    """
    Train the network on the training set.

    Parameters:
    net: The neural network model to be trained.
    train_loader (torch.utils.data.DataLoader): DataLoader for training data.
    epochs (int): Number of training epochs.
    iterations (int, optional): Number of iterations per epoch. -1 means using all iterations in the DataLoader.
     Defaults to -1.

    Returns:
    None
    """
    print(f'\ntrain for {epochs} epochs  {len(train_loader)} iterations each epoch')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    device = next(net.parameters()).device
    net.train()
    for epoch in range(epochs):
        print('Epoch', epoch)
        epoch_loss = 0.0
        optimizer.zero_grad()
        for (i, (images, labels)) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            epoch_loss += forward_batch(net, images, labels, criterion)
            images, labels = images.to('cpu'), labels.to('cpu')
            del images, labels
        optimizer.step()
        print(f'Epoch {epoch} finished with loss {epoch_loss}')


def forward_batch(net, images, labels, criterion):
    loss = criterion(net(images), labels)
    batch_loss = float(loss)
    loss.backward()
    del loss, images, labels
    return batch_loss


def train_limited_iterations(net: torch.nn.Module, num_iterations: int, train_loader: DataLoader):
    device = next(net.parameters()).device
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer.zero_grad()
    loss = 0.0
    for i in range(num_iterations):
        images, labels = next(iter(train_loader))
        images, labels = images.to(device), labels.to(device)
        loss += forward_batch(net, images, labels, criterion)
        images, labels = images.to('cpu'), labels.to('cpu')
        del images, labels
    print(f'{num_iterations} iteration(s) finished with loss {loss}')
    return loss


def test(net: torch.nn.Module, test_loader: DataLoader):
    """
    Validate the network on the entire test set.

    Parameters:
    net: The neural network model to be tested.
    test_loader (torch.utils.data.DataLoader): DataLoader for testing data.

    Returns:
    Tuple[float, float]: A tuple containing the loss and accuracy on the test set.
    """
    criterion = torch.nn.CrossEntropyLoss()
    device = next(net.parameters()).device
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in tqdm(test_loader):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print()
    print('*************************')
    print(f'Test Accuracy {accuracy}')
    print('*************************')

    return loss, accuracy


def evaluate_on_loaders(net: torch.nn.Module, loaders: List[DataLoader]) -> List[float]:
    """
    Evaluate the network on multiple data loaders.

    Parameters:
    net: The neural network model to be evaluated.
    loaders (List[torch.utils.data.DataLoader]): List of DataLoaders for evaluation.

    Returns:
    List[float]: List of accuracies on the respective data loaders.
    """
    accuracies = [test(net=net, test_loader=loader)[1] for loader in loaders]
    return accuracies
