import numpy as np

# Total voxels in a 128x128x128 3D volume: 2,097,152 voxels
total = float(128 * 128 * 128)

targets = [
    ("Whole Tumor (WT)", 0.941, 45000.0),
    ("Tumor Core (TC)", 0.918, 25000.0),
    ("Enhancing Tumor (ET)", 0.913, 18000.0),
    ("Overall / Mean", 0.924, 29333.3),
]

print(f"{'Target Sub-Region':<22} | {'Dice (DSC)':<12} | {'IoU (Jaccard)':<14} | {'MCC':<10} | {'Cohen Kappa':<12}")
print("-" * 78)

for name, dice, true_pos in targets:
    # 1. Exact mathematical IoU from Dice: IoU = Dice / (2 - Dice)
    iou = dice / (2.0 - dice)

    # 2. Typical confusion matrix elements in volumetric segmentation:
    # TP / (TP + 0.5*(FP + FN)) = Dice -> For balanced precision & recall: TP = true_pos * dice, FP = FN = true_pos * (1 - dice)
    tp = true_pos * dice
    fp = true_pos * (1.0 - dice)
    fn = fp
    tn = total - (tp + fp + fn)

    # Matthews Correlation Coefficient (MCC)
    num_mcc = tp * tn - fp * fn
    denom_mcc = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = num_mcc / denom_mcc

    # Cohen's Kappa Coefficient (kappa)
    p_o = (tp + tn) / total
    p_e = ((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)) / (total ** 2)
    kappa = (p_o - p_e) / (1.0 - p_e)

    print(f"{name:<22} | {dice*100:6.2f}% ({dice:.3f}) | {iou*100:6.2f}% ({iou:.3f}) | {mcc:6.4f}     | {kappa:6.4f}")

# Also check empirical variations across varying precision/recall asymmetries:
print("\nSensitivity Analysis (varying Precision vs Recall balance for Mean Dice = 92.4%):")
for precision_ratio in [0.90, 0.95, 1.0, 1.05, 1.10]:
    tp = 29333.3 * 0.924
    # 2*tp / (2*tp + fp + fn) = 0.924 => fp + fn = 2*tp*(1-0.924)/0.924
    err_sum = 2.0 * tp * (1.0 - 0.924) / 0.924
    fp = err_sum / (1.0 + precision_ratio)
    fn = err_sum - fp
    tn = total - (tp + fp + fn)
    num_mcc = tp * tn - fp * fn
    denom_mcc = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = num_mcc / denom_mcc
    p_o = (tp + tn) / total
    p_e = ((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)) / (total ** 2)
    kappa = (p_o - p_e) / (1.0 - p_e)
    print(f"  Precision/Recall Ratio={precision_ratio:.2f} -> MCC: {mcc:.4f}, Kappa: {kappa:.4f}")
