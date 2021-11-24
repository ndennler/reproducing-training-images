from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# data can be found from this location:
# https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
# make sure to save it somewhere you can access!
# you will want all of the aligned images and the identites txt files

class CelebAImageDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with identity markers (for finding files).
            root_dir (string): Directory with all the images that are cropped and aligned.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.identity_frame = pd.read_csv(csv_file, sep=' ', header=None)
        print(self.identity_frame)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.identity_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.identity_frame.iloc[idx, 0])
        image = io.imread(img_name)

        if self.transform:
            image= self.transform(image)

        return image

if __name__ == '__main__':

    dataset = CelebAImageDataset('/home/nathan/Downloads/identity_CelebA.txt', '/home/nathan/Downloads/img_align_celeba', transform=None)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    for image_batch in dataloader:
        print(image_batch.shape)
        break