import torch

def compute_topk_error(outputs, targets, topk=(1,3,5)):
    """
    outputs: logits of shape (batch_size, num_classes)
    targets: ground-truth labels of shape (batch_size,)
    topk: which top-k error rates to compute

    Returns:
        dict with keys: 'top1_error', 'top3_error', 'top5_error'
    """
    maxk = max(topk)
    batch_size = targets.size(0)

    # Top-k predicted classes
    _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()  # (maxk, batch_size)

    # True/False for whether each top-k prediction is correct
    correct = pred.eq(targets.view(1, -1))

    # Build output dictionary
    error_dict = {}

    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum().item()
        acc_k = correct_k / batch_size
        error_k = 1.0 - acc_k
        error_dict[f"top{k}_error"] = error_k

    return error_dict
