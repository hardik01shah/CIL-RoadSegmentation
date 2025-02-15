"""
DataLoader class for Road Segmentation dataset.
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

class RoadSegmentationDataset():
    def __init__(self, data_dir, train_size, val_size):
        self.data_dir = data_dir

        # Load the image names
        train_list = os.path.join(data_dir, 'train.txt')
        val_list = os.path.join(data_dir, 'val.txt')

        with open(train_list, 'r') as f:
            self.train_img_names = f.read().splitlines()
        with open(val_list, 'r') as f:
            self.val_img_names = f.read().splitlines()

        print(f'Found {len(self.train_img_names)} training images.')
        print(f'Found {len(self.val_img_names)} validation images.')

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

        self.images = {
            'train': train_images,
            'val': val_images
        }

        print(f'Using {train_size} training images.')
        print(f'Using {val_size} validation images.')
    
    def get_transform(self, split):
        
        # Define the transformations for the train set
        if split == 'train':
            transform = A.Compose([
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.Rotate(90),
                ToTensorV2()
            ])
        
        # Define the transformations for the validation set
        else:
            transform = A.Compose([
                ToTensorV2()
            ])
        
        return transform
    
    def get_dataset(self, split):
        img_names = self.images[split]
        transform = self.get_transform(split)
        return _Dataset(self.data_dir, img_names, transform)

class _Dataset():
    def __init__(self, data_dir, img_names, transform=None):
        self.data_dir = data_dir
        self.img_names = img_names
        self.transform = transform
        
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.data_dir, 'training/images', img_name)
        gt_path = os.path.join(self.data_dir, 'training/groundtruth', img_name)
        
        img = Image.open(img_path)
        gt = Image.open(gt_path)
        
        if self.transform:
            img = np.array(img)/255.0
            gt = np.array(gt)/255.0
            transformed = self.transform(image=img, mask=gt)
            img = transformed['image']
            gt = transformed['mask']

        return {
            'image': img,
            'gt': gt
        }

if __name__ == '__main__':
    data_dir = 'data'
    train_size = 100
    val_size = 20

    dataset = RoadSegmentationDataset(data_dir, train_size, val_size)
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

        for j in range(img.size(0)):
            img_j = img[j].permute(1, 2, 0).numpy()
            gt_j = gt[j].numpy()
            
            img_j = (img_j * 255).astype(np.uint8)
            gt_j = (gt_j * 255).astype(np.uint8)

            img_j = Image.fromarray(img_j)
            gt_j = Image.fromarray(gt_j, mode='L')

            img_j.save(f'tmp/train_img_{i}_{j}.png')
            gt_j.save(f'tmp/train_gt_{i}_{j}.png')

        if i == 0:
            break