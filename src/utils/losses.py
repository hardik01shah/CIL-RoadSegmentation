import torch
import torch.nn as nn

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