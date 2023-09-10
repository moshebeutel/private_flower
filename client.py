import flwr as fl

net = Net().to(DEVICE)
trainloader, testloader = load_data(root=args.data_path)

if __name__ == "__main__":
    SimpleNumpyClient(net=net, train_fn=train_fn, test_fn=test_fn, trainloader=trainloader,
                      testloader=testloader)
    fl.client.start_numpy_client(server_address="[::]:8080", client=client)