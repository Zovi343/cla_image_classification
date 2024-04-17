# STUDENT's UCO: 482857

# Description:
# This file should contain custom dataset class. The class should subclass the torch.utils.data.Dataset.
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torch.utils.data import random_split
import time


def compute_mean_std(data_dir):
    imgs = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.png'):
                path = os.path.join(root, file)
                imgs.append(path)

    pixels_per_channel = [[], [], []]

    for img_path in imgs:
        img = Image.open(img_path).convert('RGB')
        pixels = np.array(img).reshape(-1, 3)
        for i in range(3):
            pixels_per_channel[i].extend(pixels[:, i])

    mean = [np.mean(c) / 255 for c in pixels_per_channel]
    std = [np.std(c) / 255 for c in pixels_per_channel]

    return mean, std


def compute_mean_std_gpu(data_dir):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    img_tensors = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.png'):
                path = os.path.join(root, file)
                img = Image.open(path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to('cuda')
                img_tensors.append(img_tensor)

    all_images = torch.cat(img_tensors, dim=0)
    mean = torch.mean(all_images, dim=[0, 2, 3])
    std = torch.std(all_images, dim=[0, 2, 3])

    return mean.cpu().numpy(), std.cpu().numpy()


class SampleDataSpliter():
    def __init__(self, dataset):
        random_state = torch.get_rng_state()
        torch.manual_seed(42)

        total_size = len(dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.2 * total_size)
        test_size = total_size - train_size - val_size

        self.traindataset, self.valdataset, self.testdataset = random_split(dataset, [train_size, val_size, test_size])

        torch.set_rng_state(random_state)

    def get_train_dataset(self):
        return self.traindataset

    def get_val_dataset(self):
        return self.valdataset

    def get_test_dataset(self):
        return self.testdataset


class SampleDataset(Dataset):

    def __init__(self, data_dir="../public/data_cla_public"):
        self.data_dir = data_dir
        self.classes = ['bus', 'car', 'light', 'sign', 'truck', 'vegetation']
        self.data_info = self._get_images_info()

        # start = time.time()
        # dataset_mean, dataset_std = compute_mean_std_gpu(self.data_dir)
        # end = time.time()
        # print("Mean and Std computation took seconds: ", end - start)

        # Precomputed mean and std
        # TODO: switch back to computation of mean and std when submitting
        dataset_mean = np.array([0.28696328, 0.21672314, 0.25493133])
        dataset_std = np.array([0.19082104, 0.16210034, 0.17731088])

        print("Mean: ", dataset_mean)
        print("Std: ", dataset_std)

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_mean, std=dataset_std),
        ])

    def _get_images_info(self):
        data_info = []
        for cls_idx, cls in enumerate(self.classes):
            class_dir = os.path.join(self.data_dir, cls)
            for img_file in os.listdir(class_dir):
                if img_file.endswith('.png'):
                    img_path = os.path.join(class_dir, img_file)
                    data_info.append((img_path, cls_idx, img_file))
        return data_info

    def get_image_info(self, idx):
        return self.data_info[idx]

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        img_path, label, img_file = self.data_info[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, label, img_file


if __name__ == "__main__":
    testdataset = SampleDataset()
    _, _, img_file = testdataset[0]
    print(img_file)