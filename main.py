import argparse
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor
from time import sleep

from clients.simple_numpy_client import SimpleNumpyClient
from models.sixty_min_blitz_cnn import Net
from dataloaders.cifar10_loader import load_data
from trainers.simple_trainer import DEVICE, train, test
import flwr as fl


def get_client(net, train_fn, test_fn, trainloader, testloader):
    return SimpleNumpyClient(net=net, train_fn=train_fn, test_fn=test_fn, trainloader=trainloader,
                             testloader=testloader)


def start_client(client):
    print('start_client')
    fl.client.start_numpy_client(server_address="[::]:8080", client=client)


def start_server(n):
    print('start_server', n)
    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=3))


def start_node(arg):
    if isinstance(arg, SimpleNumpyClient):
        print('Sleep ...')
        sleep(1)
        print('Launch Client')
        start_client(arg)
    else:
        print('Launch Server')
        start_server(arg)


def main():
    parser = argparse.ArgumentParser(description="Private Federated Learning Flower")

    parser.add_argument("--data-path", type=str, default="/home/user1/datasets/cifar", help="dir path for datafolder")
    parser.add_argument("--num-clients", type=int, default="3", help="Number of clients in federation")
    args = parser.parse_args()

    net = Net().to(DEVICE)
    trainloader, testloader = load_data(root=args.data_path)
    clients = [get_client(net=net, train_fn=train, test_fn=test, trainloader=trainloader, testloader=testloader)
               for _ in range(args.num_clients)]

    nodes = [1] + clients

    with multiprocessing.Pool() as pool:
        pool.map(start_node, nodes)


if __name__ == "__main__":
    main()
