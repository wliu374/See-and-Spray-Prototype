import random
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

def read_image(x):
    img_arr = np.array(Image.open(x))
    if len(img_arr.shape) == 2:  # grayscale
        img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
    return img_arr

class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        h, w = image.shape[:2]

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

        image = image[top:top + new_h, left:left + new_w, :]
        target = target[top:top + new_h, left:left + new_w]
        return {'image': image, 'target': target}

class RandomFlip(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image, target= sample['image'], sample['target']
        do_mirror = np.random.randint(2)
        if do_mirror:
            image = cv2.flip(image, 1)
            target = cv2.flip(target, 1)
        return {'image': image, 'target': target}

class Normalize(object):

    def __init__(self, scale, mean, std):
        self.scale = np.float32(scale)
        self.mean = np.float32(mean)
        self.std = np.float32(std)

    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        image, target = image.astype('float32'), target.astype('float32')

        # pixel normalization
        image = (self.scale * image - self.mean) / self.std
        image, target = image.astype('float32'), target.astype('float32')

        return {'image': image, 'target': target}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self):
        pass

    def __call__(self, sample):
        # swap color axis
        # numpy image: H x W x C
        # torch image: C X H X W
        image, target = sample['image'], sample['target']
        image = image.transpose((2, 0, 1))
        # print(image.shape)
        target = np.expand_dims(target, axis=2)
        target = target.transpose((2, 0, 1))
        image, target = torch.from_numpy(image).float(), torch.from_numpy(target).float()
        return {'image': image, 'target': target}

class ZeroPadding(object):
    def __init__(self, psize=32):
        self.psize = psize

    def __call__(self, sample):
        psize = self.psize
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

class SoybeanWeedDataset(Dataset):
    def __init__(self, data_dir, data_list, image_dir, binary_map_dir, crop_number = 1, transform=None):
        self.data_dir = data_dir
        self.data_list = [name.split('\t') for name in open(data_list).read().splitlines()]
        self.image_dir = image_dir
        self.binary_map_dir = binary_map_dir
        self.transform = transform
        self.crop_number = crop_number
        # store images and generate ground truths
        self.images = {}
        self.targets = {}

        print(self.data_list)

    def __len__(self):
        return len(self.data_list)*self.crop_number

    def __getitem__(self, idx):
        base_idx = idx // self.crop_number
        crop_idx = idx % self.crop_number

        mask_color = [102,255,102]
        file_name = self.data_list[base_idx]
        image_array = np.array(Image.open(self.image_dir + file_name[0] + '.png'))
        binary_array = np.array(Image.open(self.binary_map_dir + file_name[0] + '.png'))
        binary_map = np.all(binary_array == mask_color, axis=-1).astype(np.uint8)

        # print(binary_array.shape)
        # plt.figure(figsize=(8, 8))
        # plt.imshow(binary_array, cmap='gray')
        # plt.title(file_name)
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

        sample['file_name'] = self.image_dir + file_name[0] + "_patch"+ str(crop_idx) +'.JPG'
        # print(sample['file_name'])
        return sample


if __name__ == '__main__':
    data_dir = "./data"
    train_dir = data_dir + "/train.txt"
    val_dir = data_dir + "/test.txt"
    image_dir = data_dir + "/Images/"
    binary_map_dir = data_dir + "/Masks/"
    image_scale = 1. / 255
    image_mean = [0.5319, 0.5560, 0.5453]
    image_std = [1, 1, 1]
    crop_size = (320, 320)


    train_set = SoybeanWeedDataset(
        data_dir=data_dir,
        data_list=train_dir,
        image_dir=image_dir,
        binary_map_dir=binary_map_dir,
        transform=transforms.Compose([
            RandomCrop(crop_size),
            RandomFlip(),
            # Normalize(scale=image_scale, std=image_std, mean=image_mean),
            ToTensor(),
            ZeroPadding(),
        ])
    )
    train_loader = DataLoader(
        train_set,
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )

    mean = 0.
    std = 0.

    for i, sample in enumerate(train_loader):
        print(i)
        image, target = sample['image'], sample['target']
        print(image.size())
        bs = image.size(0)
        image = image.view(bs, image.size(1), -1).float()
        mean += image.mean(2).sum(0)
        std += image.std(2).sum(0)


    print(len(train_loader))
    # print(mean)
    mean /= len(train_loader)
    std /= len(train_loader)
    print('mean',mean / 255.)
    print('std',std / 255.)