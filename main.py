import argparse
import os.path
import time
import torch
from dataloaders.data_loaders_factory import get_data_loaders
from models.sixty_min_blitz_cnn import Net
from multi_process_federated.federated_trainer import federated_train
from trainers.fine_tune_train import fine_tune_train, freeze_all_layers_but_last
from trainers.simple_trainer import evaluate_on_loaders

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"


# Function to parse command-line arguments
def get_args(parser):
    """
    Parse command-line arguments.

    Parameters:
    parser (argparse.ArgumentParser): Argument parser object

    Returns:
    argparse.Namespace: Parsed arguments
    """
    # Argument definitions
    # ...

    # Parse arguments
    args = parser.parse_args()
    return args


# Main function
def main():
    """
    Main function for Private Federated Learning Flower.
    """
    parser = argparse.ArgumentParser(description="Private Federated Learning Flower")

    args = get_args(parser)  # Get command-line arguments

    # Initialize the neural network model
    net = Net().to(DEVICE)

    if args.load_from:
        # Load pretrained weights
        assert os.path.exists(
            args.load_from), f'Given path for pre-trained weights does not exist. Got {args.load_from}'
        net.load_state_dict(torch.load(args.load_from))

    # Load data loaders for training and testing
    train_loader, train_loader_ood, test_loader, test_loader_ood = get_data_loaders(data_path=args.data_path,
                                                                                    batch_size=args.batch_size)

    if args.preform_pretrain:
        # Train model in a federated manner before fine tuning

        assert os.path.exists(args.saved_models_path), f'Create path for saved models. Got {args.saved_models_path}'

        federated_train(num_standard_clients=args.num_clients, net=net, test_loader=test_loader,
                        test_loader_ood=test_loader_ood, train_loader=train_loader)

        torch.save(net.state_dict(), f'{args.saved_models_path}/saved_at_{time.asctime()}.pt')
    else:
        assert args.load_from, f'Fine tune a never trained model?'

    # Evaluate the model on the provided loaders
    [base_accuracy_on_regular_data, base_accuracy_on_ood_data] = evaluate_on_loaders(net=net, loaders=[test_loader,
                                                                                                       test_loader_ood])

    if args.freeze_all_but_last:
        assert args.load_from or args.preform_pretrain, 'Freeze random weights???'
        assert args.freeze_all_but_last > 0, \
            f'Expected positive number of layers to freeze. Got {args.freeze_all_but_last}'

        freeze_all_layers_but_last(model=net, num_layers_to_freeze=args.freeze_all_but_last)

    # Fine-tune the model
    fine_tune_train(net=net,
                    train_loader_ood=train_loader_ood,
                    test_loader_ood=test_loader_ood,
                    test_loader=test_loader,
                    base_accuracy_on_regular_data=base_accuracy_on_regular_data,
                    base_accuracy_on_ood_data=base_accuracy_on_ood_data,
                    avg_orig=args.avg_orig)


# Entry point
if __name__ == "__main__":
    main()
