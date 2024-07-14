"""
Generate image list for training and validation.
Usage:
    python scripts/gen_img_list.py --data_dir='./data' --split_ratio=0.8
"""

import os
import random
import argparse
import warnings

# Set the random seed for reproducibility
random.seed(0)

def config():
    """
    Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Generate image list for training and validation.')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to the root directory of the dataset.')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='Ratio to split the dataset into training and validation sets.')
    args = parser.parse_args()
    return args

def gen_img_list(data_dir, split_ratio):
    """
    Generate image list for training and validation.
    """
    img_dir = os.path.join(data_dir, 'training/images')
    img_names = os.listdir(img_dir)

    # Check if image names are in the correct format: "satimage_x.png"
    for i in range(len(img_names)):
        if not img_names[i].startswith('satimage_') or not img_names[i].endswith('.png'):
            img_names.pop(i)
            warnings.warn(f'Incorrect image name format: {img_names[i]}. Please use the format "satimage_x.png".')

    # Check if the image names have corresponding mask images
    # If not, remove the image names without mask images
    gt_dir = os.path.join(data_dir, 'training/groundtruth')
    gt_names = os.listdir(gt_dir)
    for i in range(len(img_names)):
        if img_names[i] not in gt_names:
            img_names.pop(i)
            warnings.warn(f'No gt image found for {img_names[i]}. Removing the image name from the list.')

    # Shuffle the image names
    random.shuffle(img_names)

    # Split the image names into training and validation sets
    split_idx = int(len(img_names) * split_ratio)
    train_img_names = img_names[:split_idx]
    val_img_names = img_names[split_idx:]

    # Save the image names to text files
    train_fn = os.path.join(data_dir, 'train.txt')
    val_fn = os.path.join(data_dir, 'val.txt')
    
    with open(train_fn, 'w') as f:
        for img_name in train_img_names:
            f.write(img_name + '\n')
    
    with open(val_fn, 'w') as f:
        for img_name in val_img_names:
            f.write(img_name + '\n')

def main():
    args = config()
    gen_img_list(args.data_dir, args.split_ratio)

if __name__ == '__main__':
    main()