# STUDENT's UCO: 000000

# Description:
# This file should be used for performing inference on a network
# Usage: inference.py <path_2_dataset> <path_2_model> (<int_number_of_samples>)


import sys
import numpy as np
from dataset import SampleDataset
import os
import csv
import torch
from skimage import io
from skimage.util import img_as_ubyte
from dataset import SampleDataset, SampleDataSpliter
from torch.utils.data import DataLoader

# sample function for performing inference for a whole dataset
def infer_all(net, batch_size, dataloader, device, output_file):
    net.to(device)
    net.eval()

    # Open a file to write the predictions
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'class_id'])

        # Do not calculate the gradients
        with torch.no_grad():
            for i, (images, labels, image_files) in enumerate(dataloader):
                images = images.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                # Write each prediction to the CSV file
                for idx, (pred, actual) in enumerate(zip(predicted.cpu().numpy(), labels.numpy())):
                    writer.writerow([image_files[idx], pred])


# declaration for this function should not be changed
def inference(dataset_path, model_path, n_samples):
    # Check for available GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Computing with {}!'.format(device))

    # Load the model
    model = torch.load(model_path)
    model.eval()

    batch_size = 4

    # Initialize dataset and dataloader
    cityscape_dataset = SampleDataset(data_dir=dataset_path)
    sample_data_splitter = SampleDataSpliter(cityscape_dataset)
    testdataset = sample_data_splitter.get_test_dataset()
    testloader = DataLoader(testdataset, batch_size=batch_size, shuffle=False)

    output_dir = './output_predictions/'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'predictions.csv')

    # Perform inference
    if n_samples <= 0:
        infer_all(model, batch_size, testloader, device, output_file)
    else:
        limited_loader = DataLoader(testdataset, batch_size=1, shuffle=False)
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['filename', 'class_id'])

            with torch.no_grad():
                for i, (images, labels, image_file) in enumerate(limited_loader):
                    if i >= n_samples:
                        break
                    images = images.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    writer.writerow([image_file[0], predicted.item()])

    print(f"Predictions saved to {output_file}")

    ground_truth_dir = './ground_truth/'
    os.makedirs(ground_truth_dir, exist_ok=True)
    ground_truth_file = os.path.join(ground_truth_dir, 'ground_truth.csv')

    ground_truth_loader = DataLoader(testdataset, batch_size=1, shuffle=False)
    with open(ground_truth_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'class_id'])

        for i, (images, labels, image_file) in enumerate(ground_truth_loader):
            if n_samples > 0 and i >= n_samples:
                break
            writer.writerow([image_file[0], labels.item()])

    print(f"Ground truth saved to {ground_truth_file}")



# #### code below should not be changed ############################################################################
def get_arguments():
    if len(sys.argv) == 3:
        dataset_path = sys.argv[1]
        model_path = sys.argv[2]
        number_of_samples = 0
    elif len(sys.argv) == 4:
        try:
            dataset_path = sys.argv[1]
            model_path = sys.argv[2]
            number_of_samples = int(sys.argv[3])
        except Exception as e:
            print(e)
            sys.exit(1)
    else:
        print("Usage: inference.py <path_2_dataset> <path_2_model> (<int_number_of_samples>)")
        sys.exit(1)

    return dataset_path, model_path, number_of_samples


if __name__ == "__main__":
    path_2_dataset, path_2_model, n_samples_2_predict = get_arguments()
    inference(path_2_dataset, path_2_model, n_samples_2_predict)
