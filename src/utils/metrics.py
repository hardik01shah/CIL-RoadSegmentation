import torch
import matplotlib.pyplot as plt
import numpy as np

def segmentation_metrics(pred, gt, config, phase='train'):
    """
    Compute the segmentation metrics: Accuracy, Precision, Recall, F1 Score, and mIoU.
    
    Args:
        pred (torch.Tensor): Predicted segmentation mask.
        gt (torch.Tensor): Ground truth segmentation mask.
    
    Returns:
        dict: Segmentation metrics.
    """

    threshold = config['metrics']['threshold']
    eps = config['metrics']['eps']

    pred = (pred > threshold).float()

    tp = (pred * gt).sum().item()
    tn = ((1 - pred) * (1 - gt)).sum().item()
    fp = (pred * (1 - gt)).sum().item()
    fn = ((1 - pred) * gt).sum().item()

    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2 * precision * recall) / (precision + recall + eps)

    intersection = (pred * gt).sum().item() 
    union = (pred + gt).sum().item() - intersection
    miou = intersection / (union + eps)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'miou': miou
    }

def segmentation_metrics_eval(pred, gt, threshold=0.5, eps=1e-7):
    """
    Compute the segmentation metrics during Eval: Accuracy, Precision, Recall, F1 Score, and mIoU.
    
    Args:
        pred (numpy.ndarray): Predicted segmentation mask. Shape (H, W).
        gt (numpy.ndarray): Ground truth segmentation mask. Shape (H, W).
        threshold (float): Threshold for binarizing the prediction mask.
    """

    # Binarize the prediction mask only if threshold is provided
    if threshold is not None:
        pred = (pred > threshold).astype(np.float32)
    
    assert pred.shape == gt.shape, "Prediction and ground truth masks should have the same shape."
    assert (np.unique(pred).tolist() == [0, 1]) or (np.unique(pred).tolist() == [0]), f"Prediction mask should be binarized. Values found: {np.unique(pred).tolist()}"
    assert np.unique(gt).tolist() == [0, 1], f"Ground truth mask should be binarized. Values found: {np.unique(gt).tolist()}"

    tp = (pred * gt).sum()
    tn = ((1 - pred) * (1 - gt)).sum()
    fp = (pred * (1 - gt)).sum()
    fn = ((1 - pred) * gt).sum()

    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2 * precision * recall) / (precision + recall + eps)

    intersection = (pred * gt).sum() 
    union = (pred + gt).sum() - intersection
    miou = intersection / (union + eps)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'miou': miou
    }
    

def update_pr_curve(pred, gt, thresholds):
    """
    Compute the precision and recall values for different thresholds.
    """

    tp = torch.zeros(len(thresholds))
    tn = torch.zeros(len(thresholds))
    fp = torch.zeros(len(thresholds))
    fn = torch.zeros(len(thresholds))

    for i, threshold in enumerate(thresholds):
        pred_i = (pred > threshold).float()

        tp[i] = (pred_i * gt).sum().item()
        tn[i] = ((1 - pred_i) * (1 - gt)).sum().item()
        fp[i] = (pred_i * (1 - gt)).sum().item()
        fn[i] = ((1 - pred_i) * gt).sum().item()
    
    return tp, tn, fp, fn

def compute_pr_curve(tp, tn, fp, fn, config):
    """
    Compute the precision and recall values for different thresholds.
    """
    eps = config['metrics']['eps']
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2 * precision * recall) / (precision + recall + eps)

    return precision, recall, f1

def get_best_threshold(f1, thresholds):
    """
    Get the best threshold based on the F1 score.
    """
    best_idx = f1.argmax().item()
    best_threshold = thresholds[best_idx]

    return best_threshold, best_idx

def plot_pr_curve(precision, recall, f1, thresholds):
    """
    Plot the precision-recall curve and return the image.
    """

    # Clear the current figure
    plt.clf()

    plt.plot(recall, precision, label='Precision-Recall Curve')
    plt.plot(recall, f1, label='F1 Score')

    best_threshold, best_idx = get_best_threshold(f1, thresholds)
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]
    best_f1 = f1[best_idx]

    plt.plot(best_recall, best_precision, 'ro', label=f'Best Threshold: {best_threshold:.2f}\nF1: {best_f1:.2f}')

    # display threshold values
    for i, threshold in enumerate(thresholds):
        plt.text(recall[i], precision[i], f"{threshold:.2f}")

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()

    return plt.gcf()
    



