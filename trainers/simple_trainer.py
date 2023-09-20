import torch
from tqdm import tqdm

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"


def train(net, trainloader, epochs, iterations=-1):
    """Train the network on the training set."""
    print(f'train for {epochs} epochs {iterations if iterations > 0 else len(trainloader)} iterations each epoch')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        epoch_loss = 0.0
        for (i, (images, labels)) in enumerate(trainloader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            epoch_loss += float(loss)
            loss.backward()
            optimizer.step()
            del loss
            if 0 < iterations < i:
                break
        pbar.set_description(f'Epoch {epoch} loss {epoch_loss}')


def test(net, testloader):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
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
