# STUDENT's UCO: 482857

# Description:
# This file should be used for performing training of a network
# Usage: python training.py <path_2_dataset>

import sys
import matplotlib.pyplot as plt

import torch
from torchview import draw_graph
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import utils as vutils
from network import ModelExample
from dataset import SampleDataset, SampleDataSpliter


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

    net.to(device)

    for epoch in range(epochs):
        # Training phase
        net.train()
        running_loss = 0.0
        for data, labels, _ in trainloader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(data)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(trainloader)
        train_losses.append(avg_train_loss)

        # Validation phase
        net.eval()
        running_loss = 0.0
        with torch.no_grad():
            for data, labels, img_file in validloader:
                data, labels = data.to(device), labels.to(device)
                outputs = net(data)
                loss = loss_fn(outputs, labels)
                running_loss += loss.item()

        avg_val_loss = running_loss / len(validloader)
        validation_losses.append(avg_val_loss)
        print(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss:.5f}, Val Loss: {avg_val_loss:.5f}')

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
    epochs = 5

    cityscape_dataset = SampleDataset(data_dir=dataset_path)
    sample_data_splitter = SampleDataSpliter(cityscape_dataset)

    traindataset = sample_data_splitter.get_train_dataset()
    valdataset = sample_data_splitter.get_val_dataset()

    trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valdataset, batch_size=batch_size, shuffle=False)

    number_of_classes = 6

    net = ModelExample(number_of_classes)
    input_sample = torch.zeros((1, 3, 256, 256)).to(device)
    draw_network_architecture(net, input_sample)

    optimizer = optim.Adam(net.parameters(), lr=0.001)

    loss_fn = nn.CrossEntropyLoss()

    tr_losses, val_losses = fit(net, batch_size, epochs, trainloader, valloader, loss_fn, optimizer, device)

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
    path_2_dataset = "../public/data_cla_public"
    training(path_2_dataset)
