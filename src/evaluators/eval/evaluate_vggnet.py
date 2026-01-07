import torch
from tqdm.auto import tqdm 

from src.evaluators.metrics import compute_topk_error

def evaluate(model, dataloader, criterion, device, sanity_check=True):
    model.eval()
    pbar = tqdm(dataloader, desc="Eval ", leave=False)

    total_loss = 0.0
    total_top1_error = 0.0
    total_top3_error = 0.0
    total_top5_error = 0.0
    num_batches = 0

    # --- sanity accumulators (only for first batch) ---
    first_batch_checked = False

    with torch.no_grad():
        for features, labels in pbar:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            errors = compute_topk_error(outputs, labels, topk=(1,3,5))

            total_loss += loss.item()
            total_top1_error += errors["top1_error"]
            total_top3_error += errors["top3_error"]
            total_top5_error += errors["top5_error"]
            num_batches += 1

            # ---------------- SANITY CHECK ----------------
            if sanity_check and not first_batch_checked:
                logits_std = outputs.std().item()
                logits_mean = outputs.mean().item()
                pred_classes = outputs.argmax(dim=1)
                unique_preds = pred_classes.unique().numel()
                unique_labels = labels.unique().numel()

                print("\n[Sanity Check – Eval]")
                print(f"  Logits mean     : {logits_mean:.4f}")
                print(f"  Logits std      : {logits_std:.4f}")
                print(f"  Unique preds    : {unique_preds}")
                print(f"  Unique labels   : {unique_labels}")
                print(f"  Top-1 error     : {errors['top1_error']:.4f}")
                print("--------------------------------------------------")

                # Hard assertions (comment out once stable)
                assert logits_std > 0.01, "⚠️ Logits collapsed (near-uniform)"
                assert unique_preds > 1, "⚠️ Model predicts single class"
                #assert unique_labels > 1, "⚠️ Test batch has degenerate labels"

                first_batch_checked = True
            # ------------------------------------------------

            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

    pbar.close()

    if num_batches == 0:
        return 0, 0, 0, 0

    return (
        total_loss / num_batches,
        total_top1_error / num_batches,
        total_top3_error / num_batches,
        total_top5_error / num_batches,
    )