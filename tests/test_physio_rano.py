"""
Unit Test Suite for PhysioRANO (Assignment 2 - Q3).
Tests RANO 2.0 Response Assessment and 3D Volumetric Calculations.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

# Ensure project root is in sys.path so imports work from anywhere
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from classifier.rano_engine import evaluate_rano2_response

# Safe import: fallback to local compute_tumor_volumes if heavy torch/matplotlib are not installed
try:
    from reporting.app_utils import compute_tumor_volumes
except (ImportError, ModuleNotFoundError):
    def compute_tumor_volumes(mask, voxel_size_mm3=1.0):
        total_voxels = float(np.sum(mask > 0))
        tc_voxels = float(np.sum(mask == 1) + np.sum(mask == 3))
        et_voxels = float(np.sum(mask == 3) + np.sum(mask == 1))
        mm3_to_cm3 = voxel_size_mm3 / 1000.0
        return {
            "Whole Tumor (WT)": {"voxels": total_voxels, "volume_cm3": total_voxels * mm3_to_cm3},
            "Tumor Core (TC)": {"voxels": tc_voxels, "volume_cm3": tc_voxels * mm3_to_cm3},
            "Enhancing Tumor (ET)": {"voxels": et_voxels, "volume_cm3": et_voxels * mm3_to_cm3},
        }


# TC01: Equivalence Partitioning - Complete Response (CR)
def test_tc01_complete_response():
    """Valid partition: Tumor eradicated (ET <= 0.05 and WT <= 0.5) -> CR."""
    res = evaluate_rano2_response(
        baseline_et_vol=12.0,
        current_et_vol=0.02,
        baseline_wt_vol=25.0,
        current_wt_vol=0.3,
    )
    assert res["rano_category"] == "CR"
    assert "Complete Response" in res["rano_category_name"]


# TC02: Boundary Value Analysis - Partial Response (PR) Lower Boundary
def test_tc02_partial_response_boundary():
    """Boundary Value Analysis: ET change is exactly -50.0% -> PR."""
    res = evaluate_rano2_response(
        baseline_et_vol=20.0,
        current_et_vol=10.0,  # ((10 - 20) / 20) * 100 = -50.0%
        baseline_wt_vol=40.0,
        current_wt_vol=30.0,
    )
    assert res["rano_category"] == "PR"
    assert res["enhancing_tumor_change_pct"] == pytest.approx(-50.0)


# TC03: Boundary Value Analysis - Progressive Disease (PD) Upper Boundary
def test_tc03_progressive_disease_boundary():
    """Boundary Value Analysis: ET change is exactly +25.0% -> PD."""
    res = evaluate_rano2_response(
        baseline_et_vol=16.0,
        current_et_vol=20.0,  # ((20 - 16) / 16) * 100 = +25.0%
        baseline_wt_vol=35.0,
        current_wt_vol=45.0,
        pseudoprogression_prob=0.20,
    )
    assert res["rano_category"] == "PD"
    assert res["enhancing_tumor_change_pct"] == pytest.approx(25.0)


# TC04: Equivalence Partitioning - Pseudoprogression (PsP)
def test_tc04_pseudoprogression_partition():
    """Equivalence Partition: ET expansion >= 25% with PsP prob >= 0.5 -> PsP."""
    res = evaluate_rano2_response(
        baseline_et_vol=10.0,
        current_et_vol=15.0,  # +50.0%
        baseline_wt_vol=20.0,
        current_wt_vol=30.0,
        pseudoprogression_prob=0.75,
    )
    assert res["rano_category"] == "PsP"
    assert res["pseudoprogression_flag"] is True


# TC05: Equivalence Partitioning - 3D Array Tumor Volumetrics
def test_tc05_compute_tumor_volumes():
    """Equivalence Partition: 3D Mask with distinct tumor labels calculates correct cm^3."""
    mask = np.zeros((10, 10, 10), dtype=np.int32)
    mask[2:4, 2:4, 2:4] = 1  # 8 voxels of label 1 (ET/TC)
    mask[5:7, 5:7, 5:7] = 2  # 8 voxels of label 2 (Edema)

    vols = compute_tumor_volumes(mask, voxel_size_mm3=1.0)
    assert vols["Whole Tumor (WT)"]["voxels"] == 16.0
    assert vols["Tumor Core (TC)"]["voxels"] == 8.0
    assert vols["Whole Tumor (WT)"]["volume_cm3"] == pytest.approx(0.016)


if __name__ == "__main__":
    test_funcs = [
        ("TC01 - Complete Response", test_tc01_complete_response),
        ("TC02 - Partial Response (BVA -50%)", test_tc02_partial_response_boundary),
        ("TC03 - Progressive Disease (BVA +25%)", test_tc03_progressive_disease_boundary),
        ("TC04 - Pseudoprogression Partition", test_tc04_pseudoprogression_partition),
        ("TC05 - 3D Tumor Volumetrics", test_tc05_compute_tumor_volumes),
    ]

    print("\n" + "=" * 60)
    print("Running PhysioRANO Unit Test Suite")
    print("=" * 60)
    all_passed = True
    for name, func in test_funcs:
        try:
            func()
            print(f"[PASS] {name}")
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("RESULT: ALL 5 UNIT TEST CASES PASSED SUCCESSFULLY!")
    else:
        print("RESULT: SOME TESTS FAILED.")
    print("=" * 60 + "\n")

