import torch
import torch.nn as nn
from .loss_utils import *

def weighted_bce(pred, gt, config):
    """
    Compute the weighted binary cross entropy loss.
    
    Args:
        pred (torch.Tensor): Predicted segmentation mask.
        gt (torch.Tensor): Ground truth segmentation mask.
    
    Returns:
        torch.Tensor: Weighted binary cross entropy loss.
    """
    
    pos_weight = config['loss']['pos_weight']

    # Check shape - [N, C, H, W]
    assert pred.size() == gt.size()
    
    bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=gt * pos_weight, reduction='mean')
    loss = bce_loss(pred, gt)
    
    return loss

def patch_bce(pred, gt, config):
    """
    Compute the weighted binary cross entropy loss.
    
    Args:
        pred (torch.Tensor): Predicted segmentation mask.
        gt (torch.Tensor): Ground truth segmentation mask.
    
    Returns:
        torch.Tensor: Weighted binary cross entropy loss.
    """
    
    pos_weight = config['loss']['pos_weight']

    # Check shape - [N, C, H, W]
    assert pred.size() == gt.size()
    
    b,c,h,w = pred.shape

    pred = pred.reshape(b, 16,16,16,16).permute(0,1,3,2,4).reshape(b, 16*16,-1).mean(axis = 2)
    gt = gt.reshape(b, 16,16,16,16).permute(0,1,3,2,4).reshape(b, 16*16,-1).mean(axis = 2) > 0.25
    
    bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=gt * pos_weight, reduction='mean')
    loss = bce_loss(pred, gt*1.0)
    
    return loss

def bce_dice(pred, gt, config):
    """
    Compute the BCE Dice loss.
    
    Args:
        pred (torch.Tensor): Predicted segmentation mask.
        gt (torch.Tensor): Ground truth segmentation mask.
    
    Returns:
        torch.Tensor: Weighted binary cross entropy loss.
    """
    
    class_weight = [1, config['loss']['pos_weight']]
    loss = DiceBCELoss(Linear=True)

    # Check shape - [N, C, H, W]
    assert pred.size() == gt.size()
    
    bce_dice_loss = DiceBCELoss(Linear=True)

    loss = bce_dice_loss(pred, gt)
    
    return loss

def focal(pred, gt, config):
    """
    Compute the BCE Dice loss.
    
    Args:
        pred (torch.Tensor): Predicted segmentation mask.
        gt (torch.Tensor): Ground truth segmentation mask.
    
    Returns:
        torch.Tensor: Weighted binary cross entropy loss.
    """
    
    class_weight = [1, config['loss']['pos_weight']]
    loss = DiceBCELoss(Linear=True)

    # Check shape - [N, C, H, W]
    assert pred.size() == gt.size()
    
    focal_loss = FocalLoss(Linear=True)

    loss = focal_loss(pred, gt)
    
    return loss