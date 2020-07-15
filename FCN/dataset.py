### make a VOC dataset for segmentation

import os
import torch
import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils import data
import torchvision.datasets as datasets



root = '/home/dayun/data/'

# Download VOC dataset if not exists.
if os.path.isdir(root+'TrainVal') == False:    
    voc = datasets.VOCSegmentation(root, year='2011', download=True)

voc_root = root + 'TrainVal/VOCdevkit/VOC2011/'

# Load image and mask path from txt file
def make_dataset(mode):
    assert mode in ['train', 'val']
    items = []
    if mode == 'train':
        img_path = os.path.join(voc_root, 'JPEGImages')
        mask_path = os.path.join(voc_root, 'SegmentationClass')
        data_list = [l.strip('\n') for l in open(os.path.join(
            voc_root, 'ImageSets', 'Segmentation', 'train.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.png'))
            items.append(item)
    elif mode == 'val':
        img_path = os.path.join(voc_root, 'JPEGImages')
        mask_path = os.path.join(voc_root, 'SegmentationClass')
        data_list = [l.strip('\n') for l in open(os.path.join(
            voc_root, 'ImageSets', 'Segmentation', 'val.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.png'))
            items.append(item)

    return items


# Make mask image to label
def image_to_flat_annotation(file_name):
    img = Image.open(file_name)
    data = np.asarray( img, dtype="long" )
    return data

# define new dataset gives VOC (img, label) pair
class VOC(data.Dataset):
    def __init__(self, mode, transform=None, target_transform=None):
        self.imgs = make_dataset(mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        mask = image_to_flat_annotation(mask_path)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return img, torch.squeeze(mask, 0)

    def __len__(self):
        return len(self.imgs)