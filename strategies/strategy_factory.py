from flwr.server.strategy import FedAvg
from strategies.gep_strategy import DPFedAvgFixed


def get_strategy():
    return DPFedAvgFixed(strategy=FedAvg(), num_sampled_clients=2, clip_norm=1.0)
    # return FedAvg()