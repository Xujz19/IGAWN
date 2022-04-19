import os
import imageio
import torch
from data import common
import numpy as np
from glob import glob
import torch.utils.data as data


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


class LowLight(data.Dataset):
    def __init__(self, args, mode='train'):
        super(LowLight, self).__init__()
        self.args = args
        self.n_colors = args.n_colors
        self.mode = mode

        self._scan()

    def _scan(self):
        self.image_names = glob(self.args.train_data + '/*/Train/Normal/*.png')
        return

    def __getitem__(self, idx):
        high_name = self.image_names[idx]
        low_name = high_name.replace('Normal', 'Low')
        low_name = low_name.replace('normal', 'low')
        low_illumination_name = high_name.replace('Normal', 'Normal_illumination_by_RetinexNet')
        high = imageio.imread(high_name, pilmode='RGB')
        low = imageio.imread(low_name, pilmode='RGB')
        low_illumination = imageio.imread(low_illumination_name)

        H, W, C = high.shape
        ix = np.random.randint(0, H - self.args.patch_size + 1)
        iy = np.random.randint(0, W - self.args.patch_size + 1)

        high_patch = high[ix:ix + self.args.patch_size, iy:iy + self.args.patch_size, :]
        low_patch = low[ix:ix + self.args.patch_size, iy:iy + self.args.patch_size, :]
        low_illumination_patch = low_illumination[ix:ix + self.args.patch_size, iy:iy + self.args.patch_size, 0]

        aug_mode = np.random.randint(0, 8)
        high_patch = common.augment_img(high_patch, aug_mode)
        low_patch = common.augment_img(low_patch, aug_mode)
        low_illumination_patch = common.augment_img(low_illumination_patch, aug_mode)

        high_patch = common.image_to_tensor(high_patch)
        low_patch = common.image_to_tensor(low_patch)
        low_illumination_patch = common.image_to_tensor(low_illumination_patch)

        return low_patch, high_patch, low_illumination_patch



    def __len__(self):
        return len(self.image_names)


