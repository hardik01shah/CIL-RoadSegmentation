"""
DataLoader class for Road Segmentation dataset.
"""

import os
import numpy as np
from PIL import Image
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging

class RoadSegmentationDataset():
    def __init__(self, config):
        self.data_dir = config['data']['data_dir']
        train_size = config['data']['train_size']
        val_size = config['data']['val_size']

        # Load the image names
        train_list = os.path.join(self.data_dir, 'train.txt')
        val_list = os.path.join(self.data_dir, 'val.txt')

        mask_train_list = os.path.join(self.data_dir, 'train_mask.txt')
        mask_val_list = os.path.join(self.data_dir, 'val_mask.txt')

        with open(train_list, 'r') as f:
            self.train_img_names = f.read().splitlines()
        with open(val_list, 'r') as f:
            self.val_img_names = f.read().splitlines()

        with open(mask_train_list, 'r') as f:
            self.mask_train_img_names = f.read().splitlines()
        with open(mask_val_list, 'r') as f:
            self.mask_val_img_names = f.read().splitlines()

        logging.info(f'Found {len(self.train_img_names)} training images.')
        logging.info(f'Found {len(self.val_img_names)} validation images.')

        if train_size is None:
            train_size = len(self.train_img_names)
        if val_size is None:
            val_size = len(self.val_img_names)

        if train_size > len(self.train_img_names):
            raise ValueError(f'Train size {train_size} is greater than the number of training images {len(self.train_img_names)}.')
        if val_size > len(self.val_img_names):
            raise ValueError(f'Validation size {val_size} is greater than the number of validation images {len(self.val_img_names)}.')
        
        self.train_size = train_size
        self.val_size = val_size

        train_images = self.train_img_names[:train_size]
        val_images = self.val_img_names[:val_size]

        mask_train_images = self.mask_train_img_names[:train_size]
        mask_val_images = self.mask_val_img_names[:val_size]

        self.images = {
            'train': train_images,
            'val': val_images
        }
        self.masks = {
            'train': mask_train_images,
            'val': mask_val_images
        }

        logging.info(f'Using {train_size} training images.')
        logging.info(f'Using {val_size} validation images.')
    
    def get_transform(self, split):
        
        # Define the transformations for the train set
        if split == 'train':
            transform = A.Compose([
                # A.RandomCrop(width=256, height=256),    # TODO: Quick fix to ensure divisibility by 32 for UNet, FIX ME!
                A.Resize(width=256, height=256),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.Rotate(limit=90, p=0.5),
                A.RandomBrightnessContrast(p=0.4),
                ToTensorV2(transpose_mask=True)
            ])
        
        # Define the transformations for the validation set
        else:
            transform = A.Compose([
                A.Resize(width=256, height=256),        # TODO: Quick fix to ensure divisibility by 32 for UNet, FIX ME!
                ToTensorV2(transpose_mask=True)
            ])
        
        return transform
    
    def get_dataset(self, split):
        img_names = self.images[split]
        masks = self.masks[split]
        transform = self.get_transform(split)
        return _Dataset(self.data_dir, img_names, transform, masks)

class _Dataset():
    def __init__(self, data_dir, img_names, transform=None, masks=None):
        self.data_dir = data_dir
        self.img_names = img_names
        self.transform = transform
        self.masks = masks
        
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        mask = self.masks[idx]
        if os.path.exists(os.path.join(self.data_dir, 'training/groundtruth', img_name)):
            img_path = os.path.join(self.data_dir, 'training/images', img_name)
            gt_path = os.path.exist(os.path.join(self.data_dir, 'training/groundtruth', img_name))
        else:
            img_path = os.path.join(self.data_dir, img_name)
            gt_path = os.path.join(self.data_dir, mask)
        # print(f"Loading image: {img_path}")
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)/255.0

        gt = (cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) > 0) * 255
        # gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        assert np.unique(gt).tolist() == [0, 255] or np.unique(gt).tolist() == [0]
        gt[gt == 255] = 1
        gt = gt.astype(np.float32)
        gt = np.expand_dims(gt, axis=-1) # add channel dimension HxW -> HxWx1
        
        if self.transform:
            transformed = self.transform(image=img, mask=gt)
            img = transformed['image']
            gt = transformed['mask']

        return {
            'image': img,
            'gt': gt
        }

if __name__ == '__main__':
    config = {
        'data': {
            'data_dir': 'datasets/Full_Dataset',
            'train_size': 100,
            'val_size': 20
        }
    }

    dataset = RoadSegmentationDataset(config)
    train_dataset = dataset.get_dataset('train')
    val_dataset = dataset.get_dataset('val')

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Create tmp directory and save the transformed images for debugging
    if os.path.exists("tmp"):
        os.system("rm -r tmp")
    os.makedirs("tmp")

    for i, data in enumerate(train_loader):
        img = data['image']
        gt = data['gt']

        print(f"Image shape: {img.shape}")
        print(f"Ground truth shape: {gt.shape}")

        for j in range(img.size(0)):
            img_j = img[j].permute(1, 2, 0).numpy()
            gt_j = gt[j].permute(1, 2, 0).numpy().squeeze()
            gt_j_mask = np.stack([gt_j, gt_j, gt_j], axis = -1)

            img_j = (img_j*255).astype(np.uint8)
            o_img_j = img_j.copy()
            o_img_j[gt_j_mask == 1] = 255

            gt_j = (gt_j*255).astype(np.uint8)

            img_j = Image.fromarray(img_j)
            o_img_j = Image.fromarray(o_img_j)
            gt_j = Image.fromarray(gt_j, mode='L')

            img_j.save(f'tmp/img_{i}_{j}.png')
            o_img_j.save(f'tmp/o_img_{i}_{j}.png')
            gt_j.save(f'tmp/gt_{i}_{j}.png')

        if i == 0:
            break