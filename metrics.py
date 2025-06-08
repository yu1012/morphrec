import torch
import torch.nn.functional as F

def batched_mse(pred, gt):
    """
    Compute MSE per sample and channel.
    
    Args:
        pred (torch.Tensor): shape (N, C, T)
        gt (torch.Tensor): shape (N, C, T)
    
    Returns:
        mse (torch.Tensor): shape (N, C)
    """
    # MSE per sample per channel
    mse = F.mse_loss(pred, gt, reduction='none')  # (N, C, T)
    mse = mse.mean(dim=2)  # (N, C)

    return mse  # each is (N, C)

def batched_corr(pred, gt):
    """
    Compute correlation and MSE per sample and channel.
    
    Args:
        pred (torch.Tensor): shape (N, C, T)
        gt (torch.Tensor): shape (N, C, T)
    
    Returns:
        corr (torch.Tensor): shape (N, C)
    """
    # Correlation per sample per channel
    pred_mean = pred.mean(dim=2, keepdim=True)
    gt_mean = gt.mean(dim=2, keepdim=True)

    pred_centered = pred - pred_mean
    gt_centered = gt - gt_mean

    numerator = (pred_centered * gt_centered).sum(dim=2)
    denominator = torch.sqrt(
        (pred_centered ** 2).sum(dim=2) * (gt_centered ** 2).sum(dim=2)
    )

    corr = torch.where(denominator > 0, numerator / denominator, torch.zeros_like(numerator))

    return corr  # each is (N, C)

def build_metric():
    metrics = {
        'mse': batched_mse,
        'corr': batched_corr
    }
    return metrics
