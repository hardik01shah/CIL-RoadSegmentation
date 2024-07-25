import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

## Code referenced and edited from https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, Linear=False):
        super(DiceLoss, self).__init__()
        self.linear = Linear

    def forward(self, inputs, targets, smooth=1):
        
        if self.linear:
            inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
    
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True, Linear = False):
        super(DiceBCELoss, self).__init__()
        self.linear = Linear

    def forward(self, inputs, targets, smooth=1):
        
        if self.linear:
            inputs = F.sigmoid(inputs)          
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE
    
class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, Linear = False):
        super(TverskyLoss, self).__init__()
        self.linear = Linear

    def forward(self, inputs, targets, smooth=1, alpha=0.7, beta=0.3):
        
        if self.linear:
            inputs = F.sigmoid(inputs)
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky
    
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, Linear = False):
        super(IoULoss, self).__init__()
        self.linear = Linear

    def forward(self, inputs, targets, smooth=1):
        
        if self.linear:
            inputs = F.sigmoid(inputs)
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU
    
class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, Linear = False):
        super(FocalLoss, self).__init__()
        self.linear = Linear

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        
        if self.linear:
            inputs = F.sigmoid(inputs)
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss