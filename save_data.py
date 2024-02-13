# -*- coding: utf-8 -*-
"""
saving image data
"""

import os
# import numpy as np
# from PIL import Image
# import numbers
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
# from torchvision.transforms.functional import to_pil_image, to_tensor, _is_pil_image
from torchvision.datasets.folder import has_file_allowed_extension
# import sys
# import json
# from pprint import pprint
# from time import time
from tqdm import tqdm
import cv2
import argparse
import numpy as np
IMG_EXTENSIONS = ['.png']

class DenoisingData(torch.utils.data.Dataset):
    """Class for the denoising dataset for both train and test, with 
    file structure:
        data_root/type/noise_level/fov/capture.png
        type:           5
        noise_level:    2 (raw + ground truth)
        fov:          20 (the 19th fov is for testing)
        capture.png:    50 images in each fov --> use fewer samples

    Args:
        root (str): root directory to the dataset
        train (bool): Training set if True, else Test set
        types (seq, optional): e.g. ['TwoPhoton_BPAE_B', 'Confocal_MICE`]
        captures (int): select # images within one folder
    """
    def __init__(self, root, train, types=None, captures=50):
        #image types
        confocal_types = ['Confocal_MICE', 'Confocal_BPAE_R',
                'Confocal_BPAE_G', 'Confocal_BPAE_B', 'Confocal_FISH',]
        if types is None:
            self.types = confocal_types
        
        #number of captures
        self.captures = captures
        self.root = root
        self.train = train
        self.samples = self._gather_files()
        self.samples = self.rearrange()
        
    def _gather_files(self):
        samples = []
        #fov value
        if self.train:
            self.fov = np.arange(1,19)
        else:
            self.fov = np.arange(19,21)
        root_dir = os.path.expanduser(self.root)
        subdirs = [os.path.join(root_dir, name) for name in os.listdir(root_dir)
                if (os.path.isdir(os.path.join(root_dir, name)) and name in self.types)]
        test_mix_dir = os.path.join(root_dir, 'test_mix')
        gt_dir = os.path.join(test_mix_dir, 'gt')
        
        print(f"{subdirs=}")

        noise_dir = os.path.join(test_mix_dir, 'raw')
        print_img = True
        for subdir in tqdm(subdirs):
            gt_dir = os.path.join(subdir, 'gt')
            noise_dir = os.path.join(subdir, 'raw')
            for i_fov in tqdm(self.fov):
                noisy_fov_dir = os.path.join(noise_dir, f'{i_fov}')
                clean_file_path = os.path.join(gt_dir, f'{i_fov}', 'avg50.png')
                clean_file = []
                clean_file.append(clean_file_path)
                # print(f"{noisy_fov_dir=}; {clean_file=}")
                noisy_captures = []
                for fname in sorted(os.listdir(noisy_fov_dir))[:self.captures]:
                    if self.is_image_file(fname):
                        noisy_file = os.path.join(str(noisy_fov_dir), fname)
                        # print(f"{noisy_file=}")
                        raw = cv2.imread(noisy_file, cv2.COLOR_BGR2GRAY)
                        noisy_captures.append(raw)
                        if print_img:
                            cv2.imshow("Raw", raw)
                        
                    # randomly select one noisy capture when loading from FOV
            clean = cv2.imread(clean_file[0], cv2.COLOR_BGR2GRAY)
            if print_img:
                cv2.imshow("Clean", clean)
                cv2.waitKey(0)
            print_img = False
            samples.append((noisy_captures, clean))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def rearrange(self):
        """
        Rearranges samples into a list of (noisy, clean) pairs
        """
        new_list = []
        for class_name in self.samples:
            for noisy in class_name[0]:
                new_list.append((noisy.astype(np.float32).reshape(1, 512, 512), class_name[1].astype(np.float32).reshape(1, 512, 512)))
        return new_list


    def is_image_file(self, filename):
        """Checks if a file is an allowed image extension.
        Args:
            filename (string): path to a file
        Returns:
            bool: True if the filename ends with a known image extension
        """
        return has_file_allowed_extension(filename, IMG_EXTENSIONS)

    
def main():
    dataset = DenoisingData("/Users/emmiekao/denoising-fluorescence/denoising/dataset", True)
    # samples is a list of tuples
    # each tuple contains a list of noisy images and one corresponding clean
    # image
    for set in dataset:
        for image in set[0]:
            # image = image / 255
            image = torch.tensor(image)
        # set[1][0] = set[1][0] / 255
        set[1][0] = torch.tensor(set[1][0])

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1, drop_last=False)

if __name__ == '__main__':
    main()

# use dataloader from torch
# use torch.tensor on a numpy array to turn it into a torch tensor
# once i have the list i can just turn that into a tensor dataset
# batch size 32 (get every class represented bc i hav 12 classes)
# want to use a power of 2 in batchsize
# shuffle = true
# can try num_workers = 1 (means that once it finishes training one batch it 
# would be loading the next batch at the same time)
# drop_last = false
# divide all numpy arrays by 255, make into tensors
