import os
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import random
import numpy as np
from PIL import Image
import cv2
import scipy.io as sio
from scipy.ndimage import gaussian_filter
from skimage import util
from skimage.measure import label
from skimage.measure import regionprops
import xml.etree.ElementTree as ET
from torch.utils.data import DataLoader

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F


def read_image(x):
    # print(x)
    img_arr = np.array(Image.open(x))
    if len(img_arr.shape) == 2:  # grayscale
        img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
    return img_arr

class RandomCrop(object):
    def __init__(self, output_size, crop_num):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.crop_num = crop_num

    def __call__(self, sample):
        image_list = []
        target_list = []
        image, target = sample['image'], sample['target']
        h, w = image.shape[:2]

        for i in range(self.crop_num):
            if isinstance(self.output_size, tuple):
                new_h = min(self.output_size[0], h)
                new_w = min(self.output_size[1], w)
                assert (new_h, new_w) == self.output_size
            else:
                crop_size = min(self.output_size, h, w)
                assert crop_size == self.output_size
                new_h = new_w = crop_size

            mask = target > 0
            ch, cw = int(np.ceil(new_h / 2)), int(np.ceil(new_w / 2))
            mask_center = np.zeros((h, w), dtype=np.uint8)
            mask_center[ch:h - ch + 1, cw:w - cw + 1] = 1
            mask = (mask & mask_center)
            idh, idw = np.where(mask == 1)
            if len(idh) != 0:
                ids = random.choice(range(len(idh)))
                hc, wc = idh[ids], idw[ids]
                top, left = hc - ch, wc - cw
            else:
                top = np.random.randint(0, h - new_h + 1)
                left = np.random.randint(0, w - new_w + 1)

            image_list.append(image[top:top + new_h, left:left + new_w, :])
            target_list.append(target[top:top + new_h, left:left + new_w])

        # plt.figure(figsize=(8, 8))
        # plt.imshow(image_list[-1])
        # plt.title('random crop image')
        # plt.axis('off')
        # plt.show()
        # plt.figure(figsize=(8, 8))
        # plt.imshow(target_list[-1], cmap='gray')
        # plt.title('random crop target')
        # plt.axis('off')
        # plt.show()
        return {'image': image_list, 'target': target_list}


class RandomFlip(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image_list, target_list = sample['image'], sample['target']
        length = len(image_list)
        for i in range(length):
            do_mirror = np.random.randint(2)
            if do_mirror:
                image_list[i] = cv2.flip(image_list[i], 1)
                target_list[i] = cv2.flip(target_list[i], 1)

        # plt.figure(figsize=(8, 8))
        # plt.imshow(image_list[-1])
        # plt.title('random flip image')
        # plt.axis('off')
        # plt.show()
        # plt.figure(figsize=(8, 8))
        # plt.imshow(target_list[-1], cmap='gray')
        # plt.title('random flip target')
        # plt.axis('off')
        # plt.show()

        return {'image': image_list, 'target': target_list}


class Normalize(object):

    def __init__(self, scale, mean, std, train=True):
        self.scale = np.float32(scale)
        self.mean = np.float32(mean)
        self.std = np.float32(std)
        self.train = train

    def __call__(self, sample):
        if self.train == False:
            image, target = sample['image'], sample['target']
            image, target = image.astype('float32'), target.astype('float32')

            # pixel normalization
            image = (self.scale * image - self.mean) / self.std
            image, target = image.astype('float32'), target.astype('float32')
            return {'image': image, 'target': target}
        else:
            image_list, target_list = sample['image'], sample['target']
            length = len(image_list)
            for i in range(length):
                image_list[i], target_list[i] = image_list[i].astype("float32"), target_list[i].astype("float32")
                image_list[i] = (self.scale * image_list[i] - self.mean) / self.std
                image_list[i], target_list[i] = image_list[i].astype("float32"), target_list[i].astype("float32")

            # plt.figure(figsize=(8, 8))
            # plt.imshow(image_list[-1])
            # plt.title('normalization image')
            # plt.axis('off')
            # plt.show()
            return {'image': image_list, 'target': target_list}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, train):
        self.train = train

    def __call__(self, sample):
        if self.train == False:
            # swap color axis
            # numpy image: H x W x C
            # torch image: C X H X W
            image, target = sample['image'], sample['target']
            image = image.transpose((2, 0, 1))
            target = np.expand_dims(target, axis=2)
            target = target.transpose((2, 0, 1))
            image, target = torch.from_numpy(image), torch.from_numpy(target)
            return {'image': image, 'target': target}
        else:
            image_list, target_list = sample['image'], sample['target']
            length = len(image_list)
            for i in range(length):
                image_list[i] = image_list[i].transpose((2, 0, 1))
                target_list[i] = np.expand_dims(target_list[i], axis=2)
                target_list[i] = target_list[i].transpose((2, 0, 1))
                image_list[i], target_list[i] = torch.from_numpy(image_list[i]), torch.from_numpy(target_list[i])
            return {'image': image_list, 'target': target_list}


class ZeroPadding(object):
    def __init__(self, psize=32, train=True):
        self.psize = psize
        self.train = train

    def __call__(self, sample):
        psize = self.psize
        if self.train == False:
            image, target = sample['image'], sample['target']
            h, w = image.size()[-2:]
            ph, pw = (psize - h % psize), (psize - w % psize)
            # print(ph,pw)
            (pl, pr) = (pw // 2, pw - pw // 2) if pw != psize else (0, 0)
            (pt, pb) = (ph // 2, ph - ph // 2) if ph != psize else (0, 0)
            if (ph != psize) or (pw != psize):
                tmp_pad = [pl, pr, pt, pb]
                # print(tmp_pad)
                image = F.pad(image, tmp_pad)
                target = F.pad(target, tmp_pad)
            return {'image': image, 'target': target}
        else:
            image_list, target_list = sample['image'], sample['target']
            length = len(image_list)
            for i in range(length):
                h, w = image_list[i].size()[-2:]
                ph, pw = (psize - h % psize), (psize - w % psize)
                (pl, pr) = (pw // 2, pw - pw // 2) if pw != psize else (0, 0)
                (pt, pb) = (ph // 2, ph - ph // 2) if ph != psize else (0, 0)

                if (ph != psize) or (pw != psize):
                    tmp_pad = [pl, pr, pt, pb]
                    # print(tmp_pad)
                    image_list[i] = F.pad(image_list[i], tmp_pad)
                    target_list[i] = F.pad(target_list[i], tmp_pad)
            return {'image': image_list, 'target': target_list}


class PhragmiteDataset(Dataset):
    def __init__(self, data_dir, data_list, image_dir, binary_map_dir, phragmite_color,ratio, train=True, transform=None):
        self.data_dir = data_dir
        self.data_list = [name.split('\t') for name in open(data_list).read().splitlines()]
        self.image_dir = image_dir
        self.binary_map_dir = binary_map_dir
        self.phragmite_color = phragmite_color
        self.transform = transform
        self.train = train
        self.image_list = []
        self.ratio = ratio
        # store images and generate ground truths
        self.images = {}
        self.targets = {}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_name = self.data_list[idx]
        self.image_list.append(file_name[0])
        if file_name[0] not in self.images:
            image_array = read_image(self.image_dir + file_name[0] + '.JPG')
            binary_array = read_image(self.binary_map_dir + file_name[0] + '.png')
            # print('file_name', file_name[0])
            # print("image_array",image_array.shape)
            # print('binary_map_array',binary_array.shape)
            binary_map = np.all(binary_array == self.phragmite_color, axis=-1).astype(np.uint8)
            # print('binary_map', binary_map.shape)
            h, w = image_array.shape[:2]
            nh = int(np.ceil(h * self.ratio))
            nw = int(np.ceil(w * self.ratio))
            image_array = cv2.resize(image_array,(nw, nh), interpolation = cv2.INTER_CUBIC)
            binary_map = cv2.resize(binary_map,(nw, nh), interpolation = cv2.INTER_CUBIC)
            # plt.figure(figsize=(8, 8))
            # plt.imshow(binary_map, cmap='gray')
            # plt.title('Binary Map (Has Phragmite = 1, Others = 0)')
            # plt.axis('off')
            # plt.show()
            self.images.update({file_name[0]: image_array})
            self.targets.update({file_name[0]: binary_map})

        sample = {
            'image': self.images[file_name[0]],
            'target': self.targets[file_name[0]]
        }

        if self.transform:
            sample = self.transform(sample)

        sample['file_name'] = self.image_dir + file_name[0] + '.JPG'
        return sample


if __name__ == '__main__':

    # print([name.split('\t') for name in open('./data/train.txt').read().splitlines()])

    data_dir = "./data"
    train_dir = data_dir + "/train.txt"
    val_dir = data_dir + "/test.txt"
    image_dir = data_dir + "/Images/"
    binary_map_dir = data_dir + "/masks/"
    phragmite_color = [61, 245, 61]
    image_scale = 1. / 255
    image_mean = [0.4663, 0.4657, 0.3188]
    image_std = [1, 1, 1]
    image_mean = np.array(image_mean).reshape((1, 1, 3))
    image_std = np.array(image_std).reshape((1, 1, 3))
    crop_num = 4
    input_size = 64
    output_stride = 8

    # model-related parameters
    optimizer = 'sgd'
    batch_size = 8
    crop_size = (256, 256)
    learning_rate = 0.01
    # milestones=[200,500]
    momentum = 0.95
    mult = 1
    num_epoch = 1000
    weight_decay = 0.0005
    mae_max = 10000

    trainset = PhragmiteDataset(
        data_dir=data_dir,
        data_list=train_dir,
        image_dir=image_dir,
        binary_map_dir=binary_map_dir,
        phragmite_color=phragmite_color,
        ratio=0.25,
        train=False,
        transform=transforms.Compose([
            # RandomCrop(crop_size, crop_num),
            # RandomFlip(),
            # Normalize(scale=image_scale, std=image_std, mean=image_mean, train=True),
            ToTensor(train=False),
        ])
    )
    train_loader = DataLoader(
        trainset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )

    mean = 0.
    std = 0.
    for i, sample in enumerate(train_loader):
        image, target = sample['image'], sample['target']
        # print("len(image)",len(images))
        bs = image.size(0)
        print('image.size', image.shape)
        image = image.view(bs, image.size(1), -1).float()
        mean += image.mean(2).sum(0)
        std += image.std(2).sum(0)
        print(mean)



    print(len(train_loader))
    # print(mean)
    mean /= len(train_loader)
    std /= len(train_loader)
    print('mean',mean / 255.)
    print('std',std / 255.)
