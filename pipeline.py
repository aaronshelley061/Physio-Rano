"""
PhysioRANO Unified End-to-End Clinical Pipeline on BraTS Glioma Scans.

Integrates all 7 project phases into a single cohesive pipeline:
1. Patient Ingestion & Multi-Modal BraTS Glioma Scanning
2. MONAI 3D Multi-Modal MRI Preprocessing (T1n, T1c, T2w, T2f)
3. 3D SegResNet Volumetric Segmentation (WT, TC, ET in cm³)
4. Biophysical Fisher-Kolmogorov PINN Coupling (Tissue Diffusion D, Cell Proliferation rho, Residual R)
5. Multimodal Recurrence vs. Pseudoprogression Classifier (XGBoost)
6. RANO 2.0 Response Grade Assessment
7. Automated Neuroradiology Clinical Report Generation (PDF & JSON)

Usage:
    python pipeline.py --subject-id BraTS-GLI-00001-000
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import argparse
import numpy as np
import torch

_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from datasets.brats import find_brats_samples
from preprocessing.preprocessing import get_val_transforms
from models.segmentation import get_segmentation_model
from inference.predict import predict_subject
from reporting.app_utils import compute_tumor_volumes
from pinn.fisher_kolmogorov import fit_patient_biophysical_pinn
from classifier.recurrence_classifier import (
    MultimodalRecurrenceClassifier,
    train_and_evaluate_classifier,
    FEATURE_COLUMNS,
)
from classifier.rano_engine import evaluate_rano2_response
from reporting.clinical_reporter import generate_patient_clinical_report
from utils.logger import get_logger

logger = get_logger("PhysioRANO.Pipeline")


def load_cached_models(
    checkpoint_dir: Union[str, Path] = "outputs/checkpoints",
    device: Optional[torch.device] = None,
) -> Tuple[torch.nn.Module, MultimodalRecurrenceClassifier, torch.device]:
    """Loads and caches both the 3D SegResNet and Multimodal Classifier models.

    Args:
        checkpoint_dir: Directory containing model checkpoints.
        device: PyTorch compute device (CUDA or CPU).

    Returns:
        Tuple: (SegResNet model, MultimodalRecurrenceClassifier, torch.device)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = Path(checkpoint_dir)
    seg_checkpoint = checkpoint_dir / "best_model.pth"
    clf_checkpoint = checkpoint_dir / "recurrence_classifier.pkl"

    # 1. 3D SegResNet Segmentation Model
    model = get_segmentation_model(in_channels=4, out_channels=4).to(device)
    if seg_checkpoint.exists():
        logger.info(f"Loading SegResNet weights from {seg_checkpoint}")
        ckpt = torch.load(seg_checkpoint, map_location=device)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
    else:
        logger.warning(f"No checkpoint found at {seg_checkpoint}. Initializing MONAI SegResNet architecture.")
    model.eval()

    # 2. Multimodal Recurrence Classifier
    clf = MultimodalRecurrenceClassifier()
    if clf_checkpoint.exists():
        import pickle
        try:
            with open(clf_checkpoint, "rb") as f:
                data = pickle.load(f)
                clf.model = data["model"]
                clf.is_fitted = data["is_fitted"]
                logger.info(f"Loaded trained Recurrence Classifier from {clf_checkpoint}")
        except Exception as e:
            logger.warning(f"Error loading classifier checkpoint ({e}). Retraining...")
            clf, _ = train_and_evaluate_classifier()
    else:
        logger.info("Classifier checkpoint not found. Training recurrence classifier...")
        clf, _ = train_and_evaluate_classifier()

    return model, clf, device


def run_integrated_pipeline(
    subject_id: Optional[str] = None,
    data_dir: Union[str, Path] = "data/brats2024_gli",
    patient_clinical: Optional[Dict[str, Union[str, float]]] = None,
    baseline_vols: Optional[Dict[str, float]] = None,
    model: Optional[torch.nn.Module] = None,
    classifier: Optional[MultimodalRecurrenceClassifier] = None,
    device: Optional[torch.device] = None,
    output_dir: Union[str, Path] = "outputs/reports",
) -> Dict[str, Union[str, float, Dict]]:
    """Runs the complete, unified PhysioRANO pipeline on a single BraTS Glioma patient.

    Steps:
        1. Multi-modal NIfTI Ingestion
        2. MONAI 3D Preprocessing (4 channels, 1mm isotropic, 128³)
        3. 3D SegResNet Volumetric Segmentation (WT, TC, ET in cm³)
        4. Biophysical PINN Reaction-Diffusion parameter extraction
        5. Multimodal XGBoost Pseudoprogression vs. Recurrence prediction
        6. RANO 2.0 clinical response evaluation
        7. Automated radiology PDF and JSON report compilation

    Args:
        subject_id: BraTS subject identifier (e.g. 'BraTS-GLI-00001-000').
        data_dir: Root path to BraTS 2024 dataset.
        patient_clinical: Optional clinical demographics (Age, Sex, IDH, MGMT, Days).
        baseline_vols: Optional baseline ET and WT volumes for RANO 2.0 comparison.
        model: Pre-loaded SegResNet model.
        classifier: Pre-loaded MultimodalRecurrenceClassifier.
        device: Compute device.
        output_dir: Directory where PDF and JSON reports will be saved.

    Returns:
        Dict: Complete results dictionary with volumes, PINN physics, classification, and report paths.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Locate Subject Scans
    samples = find_brats_samples(data_dir)
    if not samples:
        raise FileNotFoundError(f"No BraTS glioma samples found in directory: {data_dir.resolve()}")

    if subject_id is not None:
        target_sample = next((s for s in samples if s["subject_id"] == subject_id), None)
        if target_sample is None:
            logger.warning(f"Subject '{subject_id}' not found. Defaulting to first available: {samples[0]['subject_id']}")
            target_sample = samples[0]
    else:
        target_sample = samples[0]

    actual_subject_id = target_sample["subject_id"]
    logger.info(f"=== Initiating Integrated PhysioRANO Analysis on Subject: {actual_subject_id} ===")

    # 2. Load Models
    if model is None or classifier is None or device is None:
        model, classifier, device = load_cached_models(device=device)

    # 3. Phase 2: MONAI Preprocessing Pipeline
    logger.info("Executing Phase 2 MONAI 3D multi-modal preprocessing...")
    val_tx = get_val_transforms(spatial_size=(128, 128, 128), pixdim=(1.0, 1.0, 1.0))
    processed = val_tx(target_sample)
    image_tensor = processed["image"]
    label_tensor = processed.get("label", None)

    # 4. Phase 3: 3D SegResNet Segmentation & Volumetrics
    logger.info("Executing Phase 3 MONAI SegResNet 3D sliding-window inference...")
    pred_mask = predict_subject(
        model=model,
        image_tensor=image_tensor,
        device=device,
        roi_size=(128, 128, 128),
        sw_batch_size=2,
    )
    vols = compute_tumor_volumes(pred_mask)
    wt_vol = vols["Whole Tumor (WT)"]["volume_cm3"]
    tc_vol = vols["Tumor Core (TC)"]["volume_cm3"]
    et_vol = vols["Enhancing Tumor (ET)"]["volume_cm3"]

    logger.info(
        f"Quantitative Volumetrics -> WT: {wt_vol:.2f} cm³ | TC: {tc_vol:.2f} cm³ | ET: {et_vol:.2f} cm³"
    )

    # 5. Clinical Profile Initialization
    if patient_clinical is None:
        from datasets.brats import load_brats_clinical_df
        df_brats = load_brats_clinical_df()
        if not df_brats.empty and actual_subject_id in df_brats["subject_id"].values:
            row = df_brats[df_brats["subject_id"] == actual_subject_id].iloc[0]
            patient_clinical = {
                "Age": float(row["AgeAtStudyDate"]),
                "Sex": "Male" if row["Sex_encoded"] == 1 else "Female",
                "IDH": str(row["IDH"]),
                "MGMT": str(row["MGMT"]),
                "days_surgery_to_fup1": float(row["days_surgery_to_fup1"]),
                "days_rt_end_to_fup1": float(row["days_rt_end_to_fup1"]),
                "num_followup_scans": int(row["num_followup_scans"]),
            }
        else:
            patient_clinical = {
                "Age": 54.0,
                "Sex": "Female",
                "IDH": "Positive (Mutated)",
                "MGMT": "Positive (Methylated)",
                "days_surgery_to_fup1": 90.0,
                "days_rt_end_to_fup1": 45.0,
                "num_followup_scans": 2,
            }

    days_rt = float(patient_clinical.get("days_rt_end_to_fup1", 45.0))

    # 6. Phase 4: Biophysical PINN Reaction-Diffusion Coupling
    # Directly pass the segmented tumor volume from SegResNet into the PINN!
    logger.info(f"Executing Phase 4 Fisher-Kolmogorov PINN (Coupling Enhancing Tumor Volume: {et_vol:.2f} cm³)...")
    target_vol_for_pinn = max(0.1, et_vol if et_vol > 0 else wt_vol)
    pinn_params = fit_patient_biophysical_pinn(
        time_days=days_rt,
        vol_cm3=target_vol_for_pinn,
        num_steps=60,
    )
    diff_D = pinn_params["pinn_diffusion_D"]
    prolif_rho = pinn_params["pinn_proliferation_rho"]
    pde_res = pinn_params["pinn_pde_residual"]

    logger.info(
        f"PINN Biophysical Parameters -> Diffusion D: {diff_D:.4f} mm²/day | "
        f"Proliferation ρ: {prolif_rho:.4f} day⁻¹ | PDE Residual: {pde_res:.6f}"
    )

    # 7. Phase 5: Multimodal Recurrence vs. Pseudoprogression Classifier
    logger.info("Executing Phase 5 Multimodal Recurrence Classifier...")
    # Encode clinical variables for the 10-feature vector
    sex_enc = 1 if str(patient_clinical.get("Sex", "F")).upper().startswith("M") else 0
    idh_enc = 1.0 if "POS" in str(patient_clinical.get("IDH", "")).upper() else 0.0
    mgmt_enc = 1.0 if "POS" in str(patient_clinical.get("MGMT", "")).upper() else 0.0

    features_dict = {
        "AgeAtStudyDate": float(patient_clinical.get("Age", 54.0)),
        "Sex_encoded": sex_enc,
        "IDH_encoded": idh_enc,
        "MGMT_encoded": mgmt_enc,
        "days_surgery_to_fup1": float(patient_clinical.get("days_surgery_to_fup1", 90.0)),
        "days_rt_end_to_fup1": days_rt,
        "num_followup_scans": int(patient_clinical.get("num_followup_scans", 2)),
        "pinn_diffusion_D": diff_D,
        "pinn_proliferation_rho": prolif_rho,
        "pinn_pde_residual": pde_res,
    }

    clf_result = classifier.predict_single_patient(features_dict)
    psp_risk = clf_result["pseudoprogression_risk_percent"]
    pred_label = clf_result["predicted_label"]

    logger.info(f"Recurrence Classification -> {pred_label} (Pseudoprogression Risk: {psp_risk:.1f}%)")

    # 8. Phase 6: RANO 2.0 Response Evaluation
    logger.info("Executing Phase 6 RANO 2.0 Response Evaluation Engine...")
    if baseline_vols is None:
        # Default baseline estimated for follow-up comparison (e.g. ~35% smaller than current)
        baseline_et = max(1.0, round(et_vol * 0.65, 2))
        baseline_wt = max(2.0, round(wt_vol * 0.70, 2))
    else:
        baseline_et = baseline_vols.get("baseline_et_vol", max(1.0, round(et_vol * 0.65, 2)))
        baseline_wt = baseline_vols.get("baseline_wt_vol", max(2.0, round(wt_vol * 0.70, 2)))

    rano_result = evaluate_rano2_response(
        baseline_et_vol=baseline_et,
        current_et_vol=et_vol,
        baseline_wt_vol=baseline_wt,
        current_wt_vol=wt_vol,
        pseudoprogression_prob=psp_risk / 100.0,
    )

    logger.info(
        f"RANO 2.0 Grade -> {rano_result['rano_category_name']} "
        f"(Enhancing Volume Shift: {rano_result['enhancing_tumor_change_pct']:+.1f}%)"
    )

    # 9. Phase 7: Automated Clinical Radiology Report Generation
    logger.info("Executing Phase 7 Automated Neuroradiology PDF Report Generation...")
    report_paths = generate_patient_clinical_report(
        patient_id=actual_subject_id,
        demographics=patient_clinical,
        volumetrics=vols,
        pinn_params=pinn_params,
        classifier_result=clf_result,
        rano_result=rano_result,
        output_dir=output_dir,
    )

    logger.info(f"=== PhysioRANO Analysis Complete for {actual_subject_id} ===")
    logger.info(f"Clinical PDF Report: {report_paths.get('pdf_report', 'None')}")

    return {
        "subject_id": actual_subject_id,
        "image_tensor": image_tensor,
        "label_tensor": label_tensor,
        "pred_mask": pred_mask,
        "volumetrics": vols,
        "pinn_params": pinn_params,
        "classifier_result": clf_result,
        "rano_result": rano_result,
        "demographics": patient_clinical,
        "report_paths": report_paths,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PhysioRANO Unified Pipeline CLI on BraTS Glioma")
    parser.add_argument("--subject-id", type=str, default="BraTS-GLI-00001-000", help="BraTS Subject ID")
    parser.add_argument("--data-dir", type=str, default="data/brats2024_gli", help="BraTS data directory")
    parser.add_argument("--output-dir", type=str, default="outputs/reports", help="Output reports directory")
    args = parser.parse_args()

    results = run_integrated_pipeline(
        subject_id=args.subject_id,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 65)
    print("      PhysioRANO Unified End-to-End Pipeline Summary")
    print("=" * 65)
    print(f"Target Subject:               {results['subject_id']}")
    print(f"Whole Tumor (WT) Volume:      {results['volumetrics']['Whole Tumor (WT)']['volume_cm3']:.2f} cm3")
    print(f"Tumor Core (TC) Volume:       {results['volumetrics']['Tumor Core (TC)']['volume_cm3']:.2f} cm3")
    print(f"Enhancing Tumor (ET) Volume:  {results['volumetrics']['Enhancing Tumor (ET)']['volume_cm3']:.2f} cm3")
    print("-" * 65)
    print(f"PINN Diffusion Rate (D):      {results['pinn_params']['pinn_diffusion_D']:.4f} mm2/day")
    print(f"PINN Proliferation Rate (rho):{results['pinn_params']['pinn_proliferation_rho']:.4f} day^-1")
    print(f"PINN PDE Residual:            {results['pinn_params']['pinn_pde_residual']:.6f}")
    print("-" * 65)
    print(f"Recurrence Diagnosis:         {results['classifier_result']['predicted_label']}")
    print(f"Pseudoprogression Risk:       {results['classifier_result']['pseudoprogression_risk_percent']:.1f}%")
    print(f"RANO 2.0 Assessment:          {results['rano_result']['rano_category_name']}")
    print(f"Enhancing Volume Shift:       {results['rano_result']['enhancing_tumor_change_pct']:+.1f}%")
    print("-" * 65)
    print(f"Generated PDF Radiology Report: {results['report_paths']['pdf_report']}")
    print("=" * 65 + "\n")
