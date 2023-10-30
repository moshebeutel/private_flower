from time import sleep
import flwr as fl
from flwr.server.strategy import FedAvg
from clients.simple_numpy_client import SimpleNumpyClient
from strategies.gep_strategy import DPFedAvgFixed


def start_client(client):
    """
    Start a client using NumPy.

    Parameters:
    client: The client to be started.

    Returns:
    None
    """
    print('start_client')
    fl.client.start_numpy_client(server_address="[::]:8080", client=client)
    print('after start client')


def start_server(num_rounds=1, strategy=FedAvg()):
    """
    Start the federated learning server.

    Returns:
    None
    """
    print('start_server')
    # fl.server.start_server(config=fl.server.ServerConfig(num_rounds=20), strategy=DPFedAvgFixed(strategy=FedAvg(),
    #                                                                                             num_sampled_clients=2,
    #                                                                                             clip_norm=1))
    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=20), strategy=strategy)
    print('after start server')


def start_node(i, *args):
    """
    Start a node, either a client or a server.

    Parameters:
    arg: An instance of SimpleNumpyClient for client, or anything else for the server.

    Returns:
    None
    """
    arg=args[i]
    if isinstance(arg, SimpleNumpyClient):
        print('Sleep ...')
        sleep(1)
        print('Launch Client')
        start_client(arg)
        print('Exit Client')
    else:
        # assert isinstance(arg, int), f'server excepts integer argument. Got {arg}'
        # assert arg > 0, f'server excepts a positive integer argument for num_rounds. Got {arg}'
        print('Launch Server')
        print(arg)
        start_server(arg[0], arg[1])
        print('Exit Server')
