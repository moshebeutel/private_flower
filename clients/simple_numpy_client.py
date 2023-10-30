import gc
from collections import OrderedDict
from copy import copy
from typing import Callable
import flwr as fl
import torch
from torch.utils.data import DataLoader

import models.model_factory


class SimpleNumpyClient(fl.client.NumPyClient):
    client_counter = 0

    # def __init__(self, net: torch.nn.Module, train_fn: Callable, test_fn: Callable,
    #              train_loader: DataLoader, test_loader: DataLoader):
    def __init__(self, train_fn: Callable, test_fn: Callable,
                 train_loader: DataLoader, test_loader: DataLoader):
        """
        Initialize a SimpleNumpyClient.

        Parameters:
        net (torch.nn.Module): The neural network model.
        train_fn (Callable): The training function for the model.
        test_fn (Callable): The testing function for the model.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        test_loader (torch.utils.data.DataLoader): DataLoader for testing data.
        """
        super(SimpleNumpyClient, self).__init__()
        SimpleNumpyClient.client_counter += 1
        self._net = None
        self._train: Callable = train_fn
        self._test: Callable = test_fn
        self._train_loader: DataLoader = train_loader
        self._test_loader: DataLoader = test_loader
        self._id: int = SimpleNumpyClient.client_counter

        print('SimpleNumpyClient.__init__', self._id)

    def get_parameters(self, config):
        """
        Get the model parameters.

        Parameters:
        config: Configuration for parameter retrieval (not used in this implementation).

        Returns:
        List[np.ndarray]: List of model parameters in numpy array format.
        """
        print()
        print('******************************************')
        print(f'SimpleNumpyClient.get_parameters id {self._id} config {config}')
        print('******************************************')
        if self._net is None:
            self._net=models.model_factory.get_model()

        parameters = [val.cpu().numpy() for _, val in self._net.state_dict().items()]
        self._net.to('cpu')
        del self._net
        gc.collect()
        torch.cuda.empty_cache()
        self._net = None

        return parameters

    def set_parameters(self, parameters):
        """
        Set the model parameters.

        Parameters:
        parameters (List[np.ndarray]): List of model parameters in numpy array format.
        """
        print()
        print('******************************************')
        print(f'SimpleNumpyClient.set_parameters id {self._id} parameters num {len(parameters)}')
        print('******************************************')
        self._net = models.model_factory.get_model()
        params_dict = zip(self._net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self._net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """
        Fit the model using the received parameters.

        Parameters:
        parameters (List[np.ndarray]): List of model parameters in numpy array format.
        config: Configuration for fitting (not used in this implementation).

        Returns:
        Tuple[List[np.ndarray], int, dict]: A tuple containing the updated model parameters,
                                           the number of samples used for training, and an empty dictionary.
        """
        print()
        print('******************************************')
        print('SimpleNumpyClient.fit', self._id)
        self.set_parameters(parameters)
        self._train(self._net.to('cuda'), self._train_loader, epochs=1)
        return self.get_parameters(config={}), len(self._train_loader), {}

    def evaluate(self, parameters, config):
        """
        Evaluate the model using the received parameters.

        Parameters:
        parameters (List[np.ndarray]): List of model parameters in numpy array format.
        config: Configuration for evaluation (not used in this implementation).

        Returns:
        Tuple[float, int, dict]: A tuple containing the loss, the number of samples used for evaluation,
                                 and a dictionary with the evaluation accuracy.
        """
        print()
        print('******************************************')
        print('SimpleNumpyClient.evaluate', self._id)
        print('******************************************')
        self.set_parameters(parameters)
        loss, accuracy = self._test(self._net.to('cuda'), self._test_loader)
        self._net.to('cpu')
        del self._net
        gc.collect()
        torch.cuda.empty_cache()
        self._net = None

        return float(loss), len(self._test_loader), {"accuracy": float(accuracy)}
