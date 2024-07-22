import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt


EPS = 1e-6

def accuracy(input, target, threshold = 0.5):

    accuracy = torch.mean((input > threshold) == target)
    return accuracy

def precision(input, target, threshold = 0.5):

    pred = input > threshold
    TP = torch.sum(pred*target)
    FP = torch.sum(pred - target > 0)

    return TP/(TP+FP + EPS)

def recall(input, target, threshold = 0.5):

    pred = input > threshold
    TP = torch.sum(pred*target)
    FN = torch.sum(target - pred > 0)

    return TP/(TP+FN + EPS)

def f1(input, target, threshold = 0.5):

    pre = precision(input, target, threshold = threshold)
    rec = recall(input, target, threshold = threshold)

    return 2*pre*rec/(pre + rec + EPS)

def IoU(input, target, threshold = 0.5):

    pred = input > threshold
    intersection = torch.sum(pred*target)
    union = torch.sum(target + pred > 0)

    return intersection/(union + EPS)

def PR_curve(input, target):

    precision, recall, thresholds = precision_recall_curve(target, input)
    plt.fill_between(recall, precision)
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.title("Train Precision-Recall curve")
    plt.savefig("PR_surve")
    
    return


## TODO, PATCH metrics based on nummber of splits
def patch_metrics(input, target, threshold = 0.5, splits = 3):

    size = input.shape[2]
    step = size//splits

    acc = []
    pre = []
    rec = []
    f1 = []
    mIoU = []

    for i in range(0, size, step):
        for j in range(0, size, step):
            accuracy = torch.mean((input > threshold) == target)

    return accuracy

