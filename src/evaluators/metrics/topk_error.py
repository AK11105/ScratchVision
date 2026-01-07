def compute_topk_error(outputs, targets, topk=(1, 3, 5)):
    """
    Compute top-k classification ERROR in percentage terms.

    Args:
        outputs (Tensor): logits of shape (B, num_classes)
        targets (Tensor): ground-truth labels of shape (B,)
        topk (tuple): which top-k error rates to compute

    Returns:
        dict with keys: 'top{k}_error' (values in %)
    """
    maxk = max(topk)
    batch_size = targets.size(0)

    # Top-k predicted class indices
    _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()  # (maxk, B)

    # Compare predictions with targets
    correct = pred.eq(targets.view(1, -1))  # (maxk, B)

    error_dict = {}

    for k in topk:
        # True if target appears anywhere in top-k
        correct_k = correct[:k].any(dim=0).float().sum().item()

        error_k = 100.0 * (1.0 - correct_k / batch_size)
        error_dict[f"top{k}_error"] = error_k

    return error_dict