# STUDENT's UCO: 000000

# Description:
# This file should be used for performing training of a network
# Usage: python training.py <path_2_dataset>

import sys
import matplotlib.pyplot as plt

import torch
from torchview import draw_graph
from network import ModelExample
from dataset import SampleDataset


# sample function for model architecture visualization
# draw_graph function saves an additional file: Graphviz DOT graph file, it's not necessary to delete it
def draw_network_architecture(network, input_sample):
    # saves visualization of model architecture to the model_architecture.png
    model_graph = draw_graph(network, input_sample, graph_dir='LR', save_graph=True, filename="model_architecture")


# sample function for losses visualization
def plot_learning_curves(train_losses, validation_losses):
    plt.figure(figsize=(10, 5))
    plt.title("Train and Evaluation Losses During Training")
    plt.plot(train_losses, label="train_loss")
    plt.plot(validation_losses, label="validation_loss")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("learning_curves.png")


# sample function for training
def fit(net, batch_size, epochs, trainloader, validloader, loss_fn, optimizer, device):
    train_losses = []
    validation_losses = []

    running_loss = 0.0
    for epoch in range(epochs):

        for batch_idx in range(0, 10):
            # add current loss
            running_loss += 0.1

            # graph variables
            train_losses.append(running_loss)
            validation_losses.append(running_loss - 0.1)

        # print training info
        print('Epoch {}, train loss: {:.5f}, val loss: {:.5f}'.format(epoch, running_loss / 42, running_loss / 42))

    print('Training finished!')
    return train_losses, validation_losses


# declaration for this function should not be changed
def training(dataset_path):
    """
    training(dataset_path) performs training on the given dataset;
    saves:
    - model.pt (trained model)
    - learning_curves.png (learning curves generated during training)
    - model_architecture.png (a scheme of model's architecture)

    Parameters:
    - dataset_path (string): path to a dataset

    Returns:
    - None
    """
    # Check for available GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Computing with {}!'.format(device))

    batch_size = 64
    traindataset, valdataset = SampleDataset(), SampleDataset()
    trainloader, valloader = None, None

    net = ModelExample()
    input_sample = torch.zeros((1, 512, 1024))
    draw_network_architecture(net, input_sample)

    # define optimizer and learning rate
    optimizer = None

    # define loss function
    loss_fn = None

    # train the network for three epochs
    tr_losses, val_losses = fit(net, batch_size, 3, trainloader, valloader, loss_fn, optimizer, device)

    # save the trained model and plot the losses, feel free to create your own functions
    torch.save(net, 'model.pt')
    plot_learning_curves(tr_losses, val_losses)
    return


# #### code below should not be changed ############################################################################
def get_arguments():
    if len(sys.argv) != 2:
        print("Usage: python training.py <path_2_dataset> ")
        sys.exit(1)

    try:
        path = sys.argv[1]
    except Exception as e:
        print(e)
        sys.exit(1)
    return path


if __name__ == "__main__":
    path_2_dataset = get_arguments()
    training(path_2_dataset)
