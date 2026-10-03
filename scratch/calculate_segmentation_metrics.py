import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import numpy as np
from datasets import get_train_val_dataloaders
from preprocessing import get_val_transforms
from models import get_segmentation_model
from inference import predict_subject

def compute_binary_metrics(y_true_bin, y_pred_bin):
    tp = np.sum((y_true_bin == 1) & (y_pred_bin == 1), dtype=np.float64)
    fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1), dtype=np.float64)
    fn = np.sum((y_true_bin == 1) & (y_pred_bin == 0), dtype=np.float64)
    tn = np.sum((y_true_bin == 0) & (y_pred_bin == 0), dtype=np.float64)

    # Dice
    denom_dice = (2.0 * tp + fp + fn)
    dice = (2.0 * tp) / denom_dice if denom_dice > 0 else (1.0 if tp + fp + fn == 0 else 0.0)

    # IoU
    denom_iou = (tp + fp + fn)
    iou = tp / denom_iou if denom_iou > 0 else (1.0 if tp + fp + fn == 0 else 0.0)

    # MCC
    denom_mcc = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom_mcc if denom_mcc > 0 else 0.0

    # Cohen's Kappa
    total = tp + fp + fn + tn
    if total > 0:
        p_o = (tp + tn) / total
        p_yes = ((tp + fp) / total) * ((tp + fn) / total)
        p_no = ((tn + fp) / total) * ((tn + fn) / total)
        p_e = p_yes + p_no
        denom_kappa = (1.0 - p_e)
        kappa = (p_o - p_e) / denom_kappa if denom_kappa > 0 else 0.0
    else:
        kappa = 0.0

    return {
        "dice": float(dice),
        "iou": float(iou),
        "mcc": float(mcc),
        "kappa": float(kappa),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = get_segmentation_model(in_channels=4, out_channels=4).to(device)
    ckpt_path = Path("outputs/checkpoints/best_model.pth")
    if ckpt_path.exists():
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    else:
        print("Checkpoint not found!")
        return

    model.eval()

    data_dir = "data/brats2024_gli"
    val_transform = get_val_transforms(spatial_size=(128, 128, 128), pixdim=(1.0, 1.0, 1.0))
    _, _, val_dataset, val_loader = get_train_val_dataloaders(
        data_dir=data_dir,
        val_transform=val_transform,
        train_ratio=0.8,
        seed=42,
    )

    print(f"Evaluating {len(val_dataset)} validation subjects...")

    wt_metrics_list = []
    tc_metrics_list = []
    et_metrics_list = []

    for i, batch in enumerate(val_loader):
        image = batch["image"]  # [B, 4, H, W, D]
        label = batch["label"].squeeze().numpy()  # [H, W, D]

        pred = predict_subject(model, image, device=device, roi_size=(128, 128, 128))

        # Whole Tumor (all non-zero labels)
        wt_true = (label > 0).astype(int)
        wt_pred = (pred > 0).astype(int)
        wt_m = compute_binary_metrics(wt_true, wt_pred)
        wt_metrics_list.append(wt_m)

        # Tumor Core (labels 1, 3 or 4)
        tc_true = np.isin(label, [1, 3, 4]).astype(int)
        tc_pred = np.isin(pred, [1, 3, 4]).astype(int)
        tc_m = compute_binary_metrics(tc_true, tc_pred)
        tc_metrics_list.append(tc_m)

        # Enhancing Tumor (labels 3 or 4)
        et_true = np.isin(label, [3, 4]).astype(int)
        et_pred = np.isin(pred, [3, 4]).astype(int)
        et_m = compute_binary_metrics(et_true, et_pred)
        et_metrics_list.append(et_m)

        print(f"Sample {i+1}/{len(val_dataset)}: WT Dice={wt_m['dice']:.4f}, IoU={wt_m['iou']:.4f}, MCC={wt_m['mcc']:.4f}, Kappa={wt_m['kappa']:.4f}")

    def mean_dict(lst):
        keys = ["dice", "iou", "mcc", "kappa"]
        return {k: float(np.mean([x[k] for x in lst])) for k in keys}

    print("\n" + "="*50)
    print("VALIDATION COHORT RESULTS (EMPIRICAL):")
    print("="*50)
    print("Whole Tumor (WT):", mean_dict(wt_metrics_list))
    print("Tumor Core (TC):", mean_dict(tc_metrics_list))
    print("Enhancing Tumor (ET):", mean_dict(et_metrics_list))

    mean_dice = np.mean([np.mean([x["dice"] for x in wt_metrics_list]), np.mean([x["dice"] for x in tc_metrics_list]), np.mean([x["dice"] for x in et_metrics_list])])
    mean_iou = np.mean([np.mean([x["iou"] for x in wt_metrics_list]), np.mean([x["iou"] for x in tc_metrics_list]), np.mean([x["iou"] for x in et_metrics_list])])
    mean_mcc = np.mean([np.mean([x["mcc"] for x in wt_metrics_list]), np.mean([x["mcc"] for x in tc_metrics_list]), np.mean([x["mcc"] for x in et_metrics_list])])
    mean_kappa = np.mean([np.mean([x["kappa"] for x in wt_metrics_list]), np.mean([x["kappa"] for x in tc_metrics_list]), np.mean([x["kappa"] for x in et_metrics_list])])

    print(f"\nOverall Mean -> Dice: {mean_dice:.4f} | IoU: {mean_iou:.4f} | MCC: {mean_mcc:.4f} | Kappa: {mean_kappa:.4f}")

if __name__ == "__main__":
    main()
