"""
Reads images from the test set and runs an ensemble of models to generate the submission file.
Usage:
    python src/eval.py --config configs/eval_base.yaml
"""

import os
import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
import argparse
import yaml
from segmentation_models_pytorch import Unet, UnetPlusPlus, DeepLabV3Plus, Linknet
from tqdm import tqdm
from utils.post_proc import *

from utils.mask_to_submission_finetune import masks_to_submission
import torch.nn.functional as nnf

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    args = parser.parse_args()
    return args

def initialize_models(config):
    models = []
    classifiers = []
    thresholds = []

    for model_config in config['models']:
        if model_config['name'] == 'unet':
            cur_model = Unet(encoder_name=model_config['backbone'])
        elif model_config['name'] == 'UnetPlusPlus':
            cur_model = UnetPlusPlus(encoder_name=model_config['backbone'])
        elif model_config['name'] == 'DeepLabV3Plus':
            cur_model = DeepLabV3Plus(encoder_name=model_config['backbone'])
        elif model_config['name'] == 'Linknet':
            cur_model = Linknet(encoder_name=model_config['backbone'])
        else:
            raise ValueError(f"Model {model_config['name']} not recognized.")
        cur_model.to(config['device'])
        cur_model.load_state_dict(torch.load(model_config['ckpt_path'])['model'])
        cur_model.eval()

        classifier = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )
        classifier.load_state_dict(torch.load(model_config['ckpt_path'])['classifier'])
        classifier.eval()

        models.append(cur_model)
        classifiers.append(classifier)
        thresholds.append(model_config['threshold'])

    return models, thresholds

def get_image_patches(img, patch_size):
    """
    Extract overlapping patches from the image that are of size patch_size x patch_size.
    """
    patches = []
    h, w, _ = img.shape
    
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            
            if i*patch_size + patch_size > h:
                i = h - patch_size
            if j*patch_size + patch_size > w:
                j = w - patch_size
            
            patch = img[i:i+patch_size, j:j+patch_size]
            patches.append(
                {
                    'patch': patch,
                    'coords': (i, j)
                }
            )

    return patches

def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize the save directory
    exp_dir = config['pred_dir_base']
    sub_name = config['submission_name']
    exp_dir = os.path.join(exp_dir, sub_name)
    pred_dir = os.path.join(exp_dir, "predictions")
    if os.path.exists(exp_dir):
        raise ValueError(f"Directory {pred_dir} already exists.")
    
    os.makedirs(pred_dir)

    submission_path = os.path.join(exp_dir, f"{sub_name}.csv")

    # initialize models
    models, classifiers, thresholds = initialize_models(config)
    
    # Load the test set
    test_dir = config['test_data_dir']
    image_filenames = [os.path.join(test_dir, name) for name in os.listdir(test_dir)]
    pred_filenames = []

    # Inference for each image on the ensemble of models
    for image_filename in tqdm(image_filenames):
        img = cv2.imread(image_filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        if config["resize"]:

            # Initialize the full prediction mask
            full_pred_mask = np.zeros((25, 25))
            img = cv2.resize(img, (256,256))
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)
            img = torch.from_numpy(img).to(config['device'])
            for model, classifier, threshold in zip(models, classifiers, thresholds):
                with torch.no_grad():
                    pred = model(img)
                    pred = nn.functional.sigmoid(pred)
                    full_pred = torch.zeros((1, 25, 25)).unsqueeze(1).to(config['device'])
                    pred = nnf.interpolate(pred, size=(400, 400), mode='bicubic', align_corners=False)
                    for i in range(25):
                        for j in range(25):
                            patch = pred[:, :, i*16:(i+1)*16, j*16:(j+1)*16]
                            patch = patch.reshape(1, -1)
                            patch = classifier(patch)
                            full_pred[:, 0, i, j] = patch.squeeze()
                    pred = full_pred

                    pred = nn.functional.sigmoid(pred)
                    pred = pred.cpu().numpy()
                    pred = pred.squeeze()
                    full_pred_mask += pred / len(models)

            # full_pred_mask =  cv2.resize(full_pred_mask, (400,400))
            # Add prediction to the full prediction mask
            full_pred_mask = (full_pred_mask > config['mean_threshold']).astype(np.uint8) 

        else:
            raise ValueError(f"Inference approach not recognized.")

        # Save the prediction mask
        pred_filename = os.path.join(pred_dir, os.path.basename(image_filename))
        pred_img = Image.fromarray((full_pred_mask)*255)
        pred_img.save(pred_filename)
        pred_filenames.append(pred_filename)

    # Generate the submission file
    # masks_to_submission(submission_path, pred_dir, *image_filenames)
    masks_to_submission(submission_path, pred_dir, *pred_filenames)

if __name__ == '__main__':
    main()