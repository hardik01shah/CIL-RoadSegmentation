"""
Reads images from the test set and uses homography adaptation to generate the submission file.
Usage:
    python src/eval_ha.py --config configs/eval_ha.yaml
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
from utils.homography import sample_homography_corners
import matplotlib.pyplot as plt

from utils.mask_to_submission import masks_to_submission

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    parser.add_argument('--viz', action='store_true', help='Visualize the predictions.')
    args = parser.parse_args()
    return args

def initialize_models(config):
    models = []
    thresholds = []
    print("Initializing models...")
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
        cur_model.load_state_dict(torch.load(model_config['ckpt_path'], map_location=config['device'])['model'])
        cur_model.eval()
        models.append(cur_model)
        thresholds.append(model_config['threshold'])
    print("Models initialized.")
    return models, thresholds

def np_img_to_tensor(img, device='cpu'):
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img).to(device)
    return img

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
    models, thresholds = initialize_models(config)

    # homography adaptation parameters
    homography_params = config['homography']
    num_H = config['num_homographies']
    pad_value = config['pad_value']
    
    # Load the test set
    test_dir = config['test_data_dir']
    image_filenames = [os.path.join(test_dir, name) for name in os.listdir(test_dir)]
    pred_filenames = []

    # Inference for each image on the ensemble of models
    for image_filename in tqdm(image_filenames):
        img = cv2.imread(image_filename)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ps = (400, 400)
        
        # pad with zeros to (448, 448)
        if pad_value > 0:
            img = cv2.copyMakeBorder(img, pad_value, pad_value, pad_value, pad_value, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            ps = (ps[0] + 2*pad_value, ps[1] + 2*pad_value)
            homography_params['patch_shape'] = [ps[0], ps[1]]

        if args.viz:
            viz_img = img.copy()
            warped_viz_imgs = []
            warped_preds = []
            inv_warped_preds = []
        
        img = img.astype(np.float32) / 255.0

        # Initialize the full prediction mask
        full_pred_mask = np.zeros((400, 400))

        # img = cv2.resize(img, (256,256))

        for model, _ in zip(models, thresholds):
            with torch.no_grad():

                for j in range(num_H):
                    if j == 0:
                        H = np.eye(3)
                    else:
                        H, _, _, _ = sample_homography_corners(img.shape[:2][::-1], **homography_params)
                    img_warped = cv2.warpPerspective(img, H, ps)

                    if args.viz:
                        warped_viz_imgs.append(cv2.warpPerspective(viz_img, H, ps))

                    img_warped = np_img_to_tensor(img_warped, device=config['device']) 
                    pred = model(img_warped)
                    pred = nn.functional.sigmoid(pred)
                    pred = pred.cpu().numpy()
                    pred = pred.squeeze()

                    if args.viz:
                        warped_preds.append((pred*255).astype(np.uint8))

                    # Warp the prediction back to the original image
                    H_inv = np.linalg.inv(H)
                    pred = cv2.warpPerspective(pred, H_inv, ps)

                    if args.viz:
                        inv_warped_preds.append((pred*255).astype(np.uint8))

                    if pad_value > 0:
                        pred = pred[pad_value:-pad_value, pad_value:-pad_value]
                    full_pred_mask += pred
        
        full_pred_mask /= (len(models) * num_H)
        # full_pred_mask =  cv2.resize(full_pred_mask, (400,400))

        # Add prediction to the full prediction mask
        full_pred_mask = (full_pred_mask > config['mean_threshold']).astype(np.uint8) 

        # Remove small artifacts and close small holes in the full_pred_mask map
        full_pred_mask = remove_small_artifacts(full_pred_mask, threshold=8)
        full_pred_mask = close_small_holes(full_pred_mask, kernel_size=16)

        if args.viz:
            # show the warped images and heatmaps for all homographies
            fig, axs = plt.subplots(4, num_H, figsize=(15, 5))
            for i in range(num_H):
                axs[0, i].imshow(warped_viz_imgs[i])
                axs[1, i].imshow(warped_preds[i])
                axs[2, i].imshow(viz_img)
                axs[3, i].imshow(inv_warped_preds[i])
                axs[0, i].axis('off')
                axs[1, i].axis('off')
                axs[2, i].axis('off')
                axs[3, i].axis('off')
            plt.show()
            plt.close()

            # Show the final prediction and the original image
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(full_pred_mask)
            axs[1].imshow(viz_img)

            axs[0].axis('off')
            axs[1].axis('off')
            plt.show()
            plt.close()

        # Save the prediction mask
        pred_filename = os.path.join(pred_dir, os.path.basename(image_filename))
        pred_img = Image.fromarray((full_pred_mask)*255)
        pred_img.save(pred_filename)
        pred_filenames.append(pred_filename)

    # Generate the submission file
    masks_to_submission(submission_path, pred_dir, *pred_filenames)

if __name__ == '__main__':
    main()