"""
saving image data
"""

import os
# import numpy as np
from PIL import Image
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
import matplotlib.pyplot as plt
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
    def __init__(self, root, train, types=None, captures=50, device='cuda:0'):
        #image types
        confocal_types = ['Confocal_MICE', 'Confocal_BPAE_R',
                'Confocal_BPAE_G', 'Confocal_BPAE_B', 'Confocal_FISH',]
        if types is None:
            self.types = confocal_types
       
        #number of captures
        self.captures = captures
        self.root = root
        self.train = train
        self.device = device
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
       

        noise_dir = os.path.join(test_mix_dir, 'raw')
        print_img = False
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
        return len(self.samples[0])

    def __getitem__(self, idx):
        return self.samples[0][idx], self.samples[1][idx]

    def rearrange(self):
        """
        Rearranges samples into a list of (noisy, clean) pairs
        """
        input_list = []
        target_list = []
        for class_name in self.samples:
            for noisy in class_name[0]:
                input_list.append(
                    torch.from_numpy(noisy.astype(np.float32).reshape(1, 512, 512))
                )
                target_list.append(
                    torch.from_numpy(class_name[1].astype(np.float32).reshape(1, 512, 512))
                )
        new_list = (input_list, target_list)
        return new_list


    def is_image_file(self, filename):
        """Checks if a file is an allowed image extension.
        Args:
            filename (string): path to a file
        Returns:
            bool: True if the filename ends with a known image extension
        """
        return has_file_allowed_extension(filename, IMG_EXTENSIONS)