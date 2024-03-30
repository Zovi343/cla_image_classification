# STUDENT's UCO: 000000

# Description:
# This file should be used for performing inference on a network
# Usage: inference.py <path_2_dataset> <path_2_model> (<int_number_of_samples>)


import sys
import numpy as np
from dataset import SampleDataset
import torch
from skimage import io
from skimage.util import img_as_ubyte


# sample function for performing inference for a whole dataset
def infer_all(net, batch_size, dataloader, device):
    # do not calculate the gradients
    with torch.no_grad():
        for i in range(0, 3):
            # generate a random image and save t to output_predictions
            random_image = np.random.rand(50, 50, 3)
            random_image_byte = img_as_ubyte(random_image)
            filename = f"output_predictions/random_image_{i}.png"  # Adjust filename as needed
            io.imsave(filename, random_image_byte)
    return


# declaration for this function should not be changed
def inference(dataset_path, model_path, n_samples):
    """
    inference(dataset_path, model_path='model.pt') performs inference on the given dataset;
    if n_samples is not passed or <= 0, the predictions are performed for all data samples at the dataset_path
    if  n_samples=N, where N is an int positive number, then only N first predictions are performed
    saves:
    - predictions to 'output_predictions' folder

    Parameters:
    - dataset_path (string): path to a dataset
    - model_path (string): path to a model
    - n_samples (int): optional parameter, number of predictions to perform

    Returns:
    - None
    """
    # Check for available GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Computing with {}!'.format(device))

    # loading the model
    model_path = 'model.pt'  # an example of model_path parameter
    model = torch.load(model_path)
    model.eval()

    batch_size = 4
    testdataset = SampleDataset()
    testloader = None

    # if n_samples <= 0 -> perform predictions all data samples at the dataset_path
    if n_samples <= 0:
        infer_all(model, batch_size, testloader, device)
    else:
        # perform predictions only for the first n_samples images
        for i in range(0, n_samples):
            # generate a random image and save t to output_predictions
            random_image = np.random.rand(50, 50, 3)
            random_image_byte = img_as_ubyte(random_image)
            filename = f"output_predictions/random_image_{i}.png"  # Adjust filename as needed
            io.imsave(filename, random_image_byte)
    return


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
