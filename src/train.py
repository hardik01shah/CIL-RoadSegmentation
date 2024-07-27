"""
Usage:
    python src/train.py --config <path_to_config_file>
Example:
    python src/train.py --config=configs/base.yaml
"""

import logging
import warnings
import yaml
import argparse
import sys
import os
import datetime
import wandb

import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from segmentation_models_pytorch import Unet, UnetPlusPlus, DeepLabV3Plus, Linknet

from train_engine import TrainEngine
import utils.losses as losses
from dataloader import RoadSegmentationDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    args = parser.parse_args()
    return args

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(config):
    
    # Set the seed
    set_seed(config['seed'])

    # Set the device
    device = config['device']
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        logging.warning('CUDA is not available. Switching to CPU.')
    logging.info(f'Using device {device}.')

    # Load the model
    if config['model']['name'] == 'unet':
        model = Unet(encoder_name = config['model']['backbone'])
    elif config['model']['name'] == 'UnetPlusPlus':
        model = UnetPlusPlus(encoder_name = config['model']['backbone'])
    elif config['model']['name'] == 'DeepLabV3Plus':
        model = DeepLabV3Plus(encoder_name = config['model']['backbone'])
    elif config['model']['name'] == 'Linknet':
        model = Linknet(encoder_name = config['model']['backbone'])
    else:
        raise NotImplementedError(f"Model {config['model']['name']} not implemented.")
    model = model.to(device)
    
    # Initialize the optimizer
    if config['train']['optimizer']['name'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['train']['optimizer']['lr'],
            weight_decay=config['train']['optimizer']['weight_decay']
        )
    else:
        raise NotImplementedError(f"Optimizer {config['train']['optimizer']['name']} not implemented.")
    
    # Initialize the loss function 
    if config['loss']['name'] == 'weighted_bce':
        criterion = losses.weighted_bce
    elif config['loss']['name'] == 'focal':
        criterion = losses.focal
    elif config['loss']['name'] == 'bce_dice':
        criterion = losses.bce_dice
    elif config['loss']['name'] == 'patch_bce':
        criterion = losses.patch_bce
    else:
        raise NotImplementedError(f"Loss function {config['loss']['name']} not implemented.")
    
    # Initialize the dataloaders
    dataset = RoadSegmentationDataset(config)
    train_dataset = dataset.get_dataset('train')
    val_dataset = dataset.get_dataset('val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=config['train']['num_workers']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['val']['batch_size'],
        shuffle=False,
        num_workers=config['val']['num_workers']
    )
    
    # Initialize the training engine
    engine = TrainEngine(
        model=model,
        device=device,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )

    # Initialize the logging
    if config["logging"]["wandb"]["id"] is not None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        wandb_id = f'{config["logging"]["wandb"]["id"]}_{timestamp}'

    else:
        wandb_id = wandb.util.generate_id()
        with open(os.path.join(config['log_dir'], "wandb_id.txt"), "w+", encoding="UTF-8") as f:
            f.write(wandb_id)
    wandb.init(
        project='road-segmentation',
        name=config['experiment_name'],
        config=config,
        entity='cil-24',
        dir=config['log_dir'],
        mode=config['logging']['wandb']['mode'],
        group=config['logging']['wandb']['group'],
        id=wandb_id
    )

    wandb.watch(model)

    # Train the model
    best_f1 = 0.0
    for epoch in range(config['train']['num_epochs']):
        engine.train_one_epoch(epoch)
        
        if epoch % config['train']['save_interval'] == 0:
            engine.save_model(config['ckpt_dir'], f'model_{epoch}.pth')

        if epoch % config['val']['val_interval'] == 0:
            cur_f1 = engine.validate()
            if cur_f1 > best_f1:
                best_f1 = cur_f1
                engine.save_model(config['ckpt_dir'], 'best_model.pth')
    
    wandb.finish()

def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup Directories
    os.makedirs(config['base_experiment_dir'], exist_ok=True)
    exp_dir = os.path.join(config['base_experiment_dir'], config['experiment_name'])
    if os.path.exists(exp_dir):
        raise FileExistsError(f"Experiment directory {exp_dir} already exists.")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'logs'), exist_ok=True)

    config['exp_dir'] = exp_dir
    config['config_file'] = args.config
    config['ckpt_dir'] = os.path.join(exp_dir, 'checkpoints')
    config['log_dir'] = os.path.join(exp_dir, 'logs')

    # Set up logging
    log_file = os.path.join(exp_dir, 'logs', 'train.log')
    logging.basicConfig(
        level=logging.INFO,   # Change this to DEBUG for more verbose logging
        format='%(asctime)s | %(levelname)8s | %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
        force=True
    )
    
    logging.info(f"Training with config file: {args.config}")
    logging.info(f"Config: \n{yaml.dump(config)}")

    # Save the config file to the experiment directory
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    # Train the model
    train(config)

if __name__ == '__main__':
    main()