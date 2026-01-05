import torch

def calculate_metrics(pred_logits, target, threshold=0.5):
    """
    Calculate Precision, Recall, F1, IoU for binary classification.
    
    Args:
        pred_logits: Model output logits (B, 1, H, W).
        target: Ground truth labels (B, 1, H, W), 0 or 1.
        threshold: Threshold for binarization after sigmoid.
        
    Returns:
        Dictionary containing precision, recall, f1, iou.
    """
    pred_prob = torch.sigmoid(pred_logits)
    pred_mask = (pred_prob > threshold).float()
    
    # Flatten tensors
    pred_mask = pred_mask.view(-1)
    target = target.view(-1)
    
    tp = (pred_mask * target).sum()
    fp = (pred_mask * (1 - target)).sum()
    fn = ((1 - pred_mask) * target).sum()
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    iou = tp / (tp + fp + fn + epsilon)
    
    return {
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'iou': iou.item()
    }

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
