import argparse
import multiprocessing
from collections import OrderedDict
from pathlib import Path
from time import sleep
import flwr as fl
import torch
from tqdm import tqdm

from clients.simple_numpy_client import SimpleNumpyClient
from dataloaders.cifar10_loader import load_data
from models.sixty_min_blitz_cnn import Net
from trainers.simple_trainer import DEVICE, train, test
import numpy as np


def get_client(net, train_fn, test_fn, trainloader, testloader):
    return SimpleNumpyClient(net=net, train_fn=train_fn, test_fn=test_fn, trainloader=trainloader,
                             testloader=testloader)


def start_client(client):
    print('start_client')
    fl.client.start_numpy_client(server_address="[::]:8080", client=client)
    print('after start client')


def start_server():
    print('start_server')
    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=20))
    print('after start server')


def start_node(arg):
    if isinstance(arg, SimpleNumpyClient):
        print('Sleep ...')
        sleep(1)
        print('Launch Client')
        start_client(arg)
        print('Exit Client')
    else:
        print('Launch Server')
        start_server()
        print('Exit Server')


def main():
    parser = argparse.ArgumentParser(description="Private Federated Learning Flower")

    parser.add_argument("--data-path", type=str, default="~/datasets/cifar", help="dir path for datafolder")
    parser.add_argument("--num-clients", type=int, default="1", help="Number of clients in federation")
    parser.add_argument("--batch-size", type=int, default="128", help="Number of images in train batch")
    parser.add_argument("--node-type", type=str, choices=['client', 'server'], default='client',
                        help='client node or server node')
    parser.add_argument("--ood", type=bool, default=False, help='client node or server node')

    args = parser.parse_args()

    net = Net().to(DEVICE)
    root = args.data_path
    root = root.replace('~', str(Path.home()))

    trainloader, testloader = load_data(root=root, batch_size=args.batch_size, ood=False)
    trainloader_ood, testloader_ood = load_data(root=root, batch_size=args.batch_size, ood=True)

    clients = [get_client(net=net, train_fn=train, test_fn=test, trainloader=trainloader, testloader=testloader)
               for _ in range(args.num_clients)]
    client_ood = get_client(net=net, train_fn=train, test_fn=test, trainloader=trainloader, testloader=testloader_ood)

    nodes = [1] + clients + [client_ood]

    with multiprocessing.Pool() as pool:
        pool.map(start_node, nodes)

    print()
    print('******************')
    print('Few shot training')
    print('******************')
    parameters = client_ood.get_parameters(config={})
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

    print()
    print('Verify weights')
    loss, acc = test(net=net, testloader=testloader)
    loss_ood, acc_ood = test(net=net, testloader=testloader_ood)
    losses_on_original, accs_on_original, losses_on_aug, accs_on_aug = [loss_ood], [acc], [loss_ood], [acc_ood]
    print(f'Retrain')
    for i in tqdm(range(len(trainloader_ood))):
        # num_few_shots = (i+1) * args.batch_size
        # print(f'train using {num_few_shots} images. ({i+1} batches of {args.batch_size})')
        train(net=net, trainloader=trainloader_ood, epochs=1, iterations=1)
        # print('Original test set. Expect degraded acc')
        loss, acc = test(net=net, testloader=testloader)
        losses_on_original.append(loss)
        accs_on_original.append(acc)
        # print('OOD test set. Expect better acc than earlier')
        loss, acc = test(net=net, testloader=testloader_ood)
        losses_on_aug.append(loss)
        accs_on_aug.append(acc)
        with open('accs_on_aug.npy', 'wb') as f:
            np.save(f, np.array(accs_on_aug))
        with open('losses_on_aug.npy', 'wb') as f:
            np.save(f, np.array(losses_on_aug))
        with open('accs_on_original.npy', 'wb') as f:
            np.save(f, np.array(accs_on_original))
        with open('losses_on_original.npy', 'wb') as f:
            np.save(f, np.array(losses_on_original))

if __name__ == "__main__":
    main()
