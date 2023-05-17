import random

import cv2, time
import tifffile
import torch
import skimage.color as sc
from data import common
import imageio
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.io import loadmat
class dataloader(data.Dataset):
    def __init__(self, args):
        self.args = args
        self._set_filesystem()

        if self.args.store_in_ram:
            self.img_HR, self.img_LR = [], []
            with tqdm(total=len(self.filepath_HR), ncols=224) as pbar:
                for idx in range(len(self.filepath_HR)):
                    img_HR, img_LR = imageio.imread(self.filepath_HR[idx]), imageio.imread(self.filepath_LR[idx])
                    self.img_HR.append(img_HR)
                    self.img_LR.append(img_LR)
                    time.sleep(0.01)
                    pbar.update(1)
                    pbar.set_postfix(name=self.filepath_HR[idx].split('/')[-1])
            self.n_train = len(self.img_HR)

    def _set_filesystem(self):
        self.filepath_HR = np.array([])
        self.filepath_LR = np.array([])
        for idx_dataset in range(len(self.args.data_train)):
            if self.args.n_train[idx_dataset] > 0:
                path = self.args.dir_data + 'Train/' + self.args.data_train[idx_dataset]
                names_HR = os.listdir(os.path.join(path, 'HR'))
                task_note = 'LR_bicubic/X{}'.format(self.args.scale)
                names_LR = os.listdir(os.path.join(path, task_note))

                names_HR.sort(), names_LR.sort()
                filepath_HR, filepath_LR = np.array([]), np.array([])

                for idx_image in range(len(names_HR)):
                    filepath_HR = np.append(filepath_HR, os.path.join(path + '/HR', names_HR[idx_image]))
                    filepath_LR = np.append(filepath_LR, os.path.join(path + '/' + task_note, names_LR[idx_image]))

                data_length = len(filepath_HR)
                idx = np.arange(0, data_length)
                if self.args.n_train[idx_dataset] < data_length:
                    if self.args.shuffle:
                        idx = np.random.choice(idx, size=self.args.n_train[idx_dataset])
                    else:
                        idx = np.arange(0, self.args.n_train[idx_dataset])

                self.filepath_HR = np.append(self.filepath_HR, filepath_HR[idx])
                self.filepath_LR = np.append(self.filepath_LR, filepath_LR[idx])

    def __getitem__(self, idx):

        if self.args.store_in_ram:
            idx = idx % len(self.img_HR)
            img_HR = self.img_HR[idx]
            img_LR = self.img_LR[idx]
        else:
            raise InterruptedError

        img_LR, img_HR = common.set_channel([img_LR, img_HR], self.args.n_colors)
        img_LR, img_HR = common.get_patch([img_LR, img_HR], self.args.patch_size, self.args.scale)
        flag_aug = random.randint(0, 7)
        img_LR = common.augment(img_LR, flag_aug)
        img_HR = common.augment(img_HR, flag_aug)
        img_LR = common.np2Tensor(img_LR, self.args.value_range)
        img_HR = common.np2Tensor(img_HR, self.args.value_range)

        return img_LR, img_HR

    def __len__(self):
        return self.args.iter_epoch * self.args.batch_size
