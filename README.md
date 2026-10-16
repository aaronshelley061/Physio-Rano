[![CI](https://github.com/aaronshelley061/Physio-Rano/actions/workflows/ci.yml/badge.svg)](https://github.com/aaronshelley061/Physio-Rano/actions/workflows/ci.yml)


# 🧠 PhysioRANO: Physiological & Physics-Informed Neuro-Oncology Framework

PhysioRANO is a PyTorch and MONAI-based research framework designed for neuro-oncology medical imaging analysis (BraTS 2024 GLI and longitudinal Burdenko datasets), 3D tumor segmentation, physics-informed biophysical growth modeling, explainable AI (XAI), and automated RANO 2.0 clinical response assessment.

---
### 👥 Team Members
| Roll No | Name | Role & Contribution |
| :--- | :--- | :--- |
| B23CS2115 | ANNA NEENA BRIGHT (Lead) | Pipeline, RANO 2.0 Engine, Unit Testing |
| B23CS2147 | NANDANA B PRASANTH | Biophysical PINN, Recurrence Classifier |
| B23CS2101 | AARON SHELLY KOSHY | Streamlit UI, CI/CD Pipeline & Deployment |

### 👥 Team Members
| Roll No | Name | Role & Contribution |
| :--- | :--- | :--- |
| <B23CS2115> | <ANNA NEENA BRIGHT> (Lead) | Pipeline, RANO 2.0 Engine, Unit Testing |
| <B23CS2147> | <NANDANA B PRASANTH> | Biophysical PINN, Recurrence Classifier |
| <B23CS2101> | <AARON SHELLY KOSHY> | Streamlit UI, CI/CD Pipeline & Deployment |

---

## 🏗️ System Architecture Overview

```
[ Block 1: Data Ingestion ] ➔ [ Block 2: Preprocessing ] ➔ [ Block 3: 3D AI Segmentation ]
  (BraTS & Burdenko)           (Resampling & Normalization)     (MONAI SegResNet 3D)
                                                                       │
┌──────────────────────────────────────────────────────────────────────┴──────────────────────────────────────────────────┐
│                                     Block 4: Multi-Branch Feature & Physics Modeling                                   │
│  ├─ Branch 4A: Feature Extraction (Volumes & Radiomics)                                                                 │
│  ├─ Branch 4B: Temporal Growth Modeling (ConvLSTM)                                                                      │
│  └─ Branch 4C: Biophysical Growth Modeling (Fisher-Kolmogorov PINN) ➔ Physics-Informed Growth Residuals                  │
└──────────────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                                       ▼
[ Block 8: RANO 2.0 Engine ] ◄── [ Block 7: SHAP & Grad-CAM ] ◄── [ Block 5 & 6: Feature Fusion & XGBoost Classifier ]
  (CR, PR, SD, PD Assessment)       (Explainability & XAI)           (True Progression vs. Pseudoprogression)
         │
         ▼
[ Output: Automated Clinical PDF & JSON Reports ]
```

---

## 🛠️ Technologies Used

The PhysioRANO platform integrates modern deep learning, biophysical mathematical modeling, machine learning, and full-stack web technologies:

| Category | Technology | Purpose & Description |
| :--- | :--- | :--- |
| **Deep Learning & Medical Imaging** | **PyTorch (>=2.0)** | Core tensor computation engine with CUDA acceleration, autograd, and Automatic Mixed Precision (AMP). |
| | **MONAI (>=1.3)** | Medical Open Network for AI; provides 3D `SegResNet` architecture, `SlidingWindowInferer`, and dictionary transforms (`Spacingd`, `CropForegroundd`, `NormalizeIntensityd`, `ConcatItemsd`). |
| | **NiBabel (>=5.1)** | Standard neuroimaging library for I/O handling of 3D multi-modal NIfTI (`.nii.gz`) MRI volumes. |
| | **SciPy & NumPy** | High-performance numerical operations, spatial gradients, and finite-difference reaction-diffusion operators. |
| | **Pandas** | Tabular clinical and demographic data ingestion (IDH mutation, MGMT promoter status, survival metrics). |
| **Biophysical PINN Modeling** | **Physics-Informed PDEs** | 3D Fisher-Kolmogorov reaction-diffusion biophysical equation ($\frac{\partial u}{\partial t} = \nabla \cdot (D \nabla u) + \rho u (1 - u)$). |
| | **Parameter Estimation** | Patient-specific estimation of tumor cell diffusion ($D$) and proliferation ($\rho$) to compute physics-informed growth residuals. |
| **Machine Learning & Analytics** | **XGBoost (>=1.7)** | Gradient-boosted decision trees for recurrence diagnosis (True Progression vs. Pseudoprogression / Radiation Necrosis). |
| | **Scikit-Learn (>=1.3)** | Stratified K-Fold cross-validation, ROC-AUC metric computation, confusion matrix evaluation, and feature normalization. |
| **Longitudinal & Generative AI** | **ConvLSTM** | Spatio-temporal recurrent neural network modeling tumor volume trajectories across longitudinal scan sessions. |
| | **3D Conditional GAN** | Generative adversarial network for 6-month prospective 3D tumor morphology and volumetric forecasting. |
| **Explainable AI (XAI)** | **SHAP (>=0.42)** | Shapley Additive exPlanations (`TreeExplainer`) providing global feature importance rankings and local patient attribution. |
| | **3D Grad-CAM** | Gradient-weighted Class Activation Mapping generating spatial 3D activation heatmaps across SegResNet layers. |
| **Clinical Decision Support** | **RANO 2.0 Engine** | Automated rules engine categorizing clinical disease response into Complete Response (CR), Partial Response (PR), Stable Disease (SD), or Progressive Disease (PD). |
| | **ReportLab (>=3.6)** | Automated generation of publication-grade neuroradiology clinical summary PDF reports with embedded multi-modal slice panels. |
| **Web Dashboards & APIs** | **Full-Stack React** | Modern React 18 browser dashboard (`frontend/app.jsx`) with Tailwind CSS, Lucide icons, and interactive multi-axial MRI viewer. |
| | **Streamlit (>=1.28)** | Interactive multi-tab clinical research dashboard (`app.py`) for live slice exploration, parameter tuning, and report downloads. |
| | **Python REST API** | Multi-threaded HTTP server (`api.py`) exposing RESTful endpoints for all 7 project phases and serving the React frontend. |
| **DevOps, Config & Testing** | **Hydra & OmegaConf** | Hierarchical YAML configuration system (`configs/config.yaml`) for reproducible experiments. |
| | **PyTest (>=7.0)** | Automated unit and integration testing suite verifying RANO classification, volumetrics, and pipeline integrity. |
| | **Hugging Face Hub** | Automated remote dataset downloading and caching for BraTS 2024 GLI cohorts. |
| | **TensorBoard & Rich** | Training telemetry visualization and styled CLI terminal logging. |

---

## 🗺️ Phase-by-Phase Implementation Roadmap

### 📍 Phase 1: Project Foundation & System Architecture *(Completed)*
* **Objective**: Build a modular, scalable project framework with configuration management and reproducible pipeline entry points.
* **Key Deliverables**:
  * **Hydra Configuration Engine**: Hierarchical YAML configuration (`configs/config.yaml`, `configs/dataset/`) for flexible experiment setup.
  * **Modular Repository Architecture**: Organized directory structure decoupling data loading, modeling, training, inference, and reporting.
  * **Logging & Seeds**: Deterministic seed utility and structured color logging across all execution modules.

### 📍 Phase 2: Multi-Modal Data Ingestion & Preprocessing Pipeline *(Completed)*
* **Objective**: Ingest multi-modal MRI sequences and apply MONAI 3D dictionary-based preprocessing transformations.
* **Key Deliverables**:
  * **BraTS 2024 GLI Auto-Ingestion**: `BraTSDataset` automatically pairs 4 MRI modalities (`T1n`, `T1c`, `T2w`, `T2f`) + ground truth segmentation mask (`seg.nii.gz`) with automated Hugging Face download integration.
  * **Longitudinal Cohort Data Structure**: `BurdenkoLongitudinalDataset` template for multi-timepoint patient scan series.
  * **3D MONAI Transforms**:
    * Isotropic voxel spatial resampling (`Spacingd` to 1.0mm³).
    * Non-zero brain tissue foreground cropping (`CropForegroundd`).
    * Channel-wise Z-score intensity normalization (`NormalizeIntensityd`).
    * Modality tensor stacking into 4-channel 3D volumes (`ConcatItemsd` $\to$ `[4, H, W, D]`).

### 📍 Phase 3: 3D AI Tumor Segmentation & Volumetric Extraction *(Completed)*
* **Objective**: Develop deep learning 3D segmentation models to identify and quantify glioblastoma sub-regions.
* **Key Deliverables**:
  * **MONAI SegResNet 3D Architecture**: 3D convolutional neural network optimized for multi-modal brain tumor segmentation.
  * **Sliding-Window Inference**: `SlidingWindowInferer` engine for memory-efficient dense 3D volume prediction.
  * **Tumor Sub-Region Quantification**: Extraction of physical volumes ($cm^3$) and voxel counts for:
    * Whole Tumor (WT)
    * Tumor Core (TC)
    * Enhancing Tumor (ET)
    * Edema (ED)

### 📍 Phase 4: Multi-Branch Feature Analysis & Biophysical Growth Modeling *(In Progress)*
* **Objective**: Analyze temporal tumor trajectory by combining radiomic features, temporal deep learning, and physics-informed neural networks.
* **Key Deliverables**:
  * **Branch 4A (Radiomics & Volumetrics)**: Extraction of PyRadiomics texture, shape, and intensity features from segmented sub-regions.
  * **Branch 4B (Temporal ConvLSTM)**: ConvLSTM network for spatio-temporal modeling of longitudinal follow-up scan series.
  * **Branch 4C (Fisher-Kolmogorov PINN)**: Physics-Informed Neural Network (PINN) enforcing reaction-diffusion PDEs ($\frac{\partial u}{\partial t} = D \nabla^2 u + \rho u (1 - u)$) to model biophysical tumor invasion and compute physics-informed growth residuals.

### 📍 Phase 5: Multimodal Feature Fusion & Recurrence Classifier *(Planned)*
* **Objective**: Differentiate **True Tumor Progression** from **Pseudoprogression** (radiation necrosis) by fusing imaging features with clinical parameters.
* **Key Deliverables**:
  * **Multimodal Fusion Layer**: Merging radiomics, ConvLSTM embeddings, PINN growth residuals, and patient clinical/molecular data (IDH status, MGMT promoter methylation, age).
  * **XGBoost Recurrence Classifier**: Supervised machine learning model outputting calibrated probabilities for True Progression vs. Pseudoprogression.

### 📍 Phase 6: Explainable AI (XAI) & Interpretability *(Planned)*
* **Objective**: Provide visual and quantitative explanations for clinical decision support.
* **Key Deliverables**:
  * **3D Grad-CAM**: Spatial heatmaps illustrating regional 3D MRI activation areas driving SegResNet model predictions.
  * **SHAP (SHapley Additive exPlanations)**: Global and local feature attribution ranking clinical and radiomic predictors in the XGBoost classifier.

### 📍 Phase 7: RANO 2.0 Engine & Clinical Reporting *(Planned / Dashboard Live)*
* **Objective**: Automate clinical response evaluation based on updated RANO 2.0 (Response Assessment in Neuro-Oncology) guidelines and generate exportable reports.
* **Key Deliverables**:
  * **RANO 2.0 Rules Engine**: Automated categorical classification into:
    * **CR** (Complete Response)
    * **PR** (Partial Response)
    * **SD** (Stable Disease)
    * **PD** (Progressive Disease)
  * **Automated Clinical Reporting**: Generation of downloadable PDF clinical summaries and structured JSON metric files.
  * **Interactive Streamlit Web Dashboard**: Live multi-tab UI (`app.py`) for visual slice inspection, model inference, volumetrics, and training analytics.

---

## 📁 Repository Layout

```
PhysioRANO/
├── configs/                # Hydra YAML configuration files
│   ├── config.yaml         # Main configuration hierarchy
│   └── dataset/            # Dataset configs (brats.yaml, burdenko.yaml)
├── datasets/               # MONAI Dataset loaders & HF downloader
│   ├── brats.py            # BraTS 2024 GLI dataset & loader
│   ├── burdenko.py         # Burdenko longitudinal dataset template
│   └── downloader.py       # Automated Hugging Face downloader
├── models/                 # Neural network architectures (SegResNet 3D, cGAN)
├── preprocessing/          # MONAI 4-modal transforms, Z-score, & resampling
├── inference/              # 3D Sliding-window inferer & NIfTI prediction
├── longitudinal/           # Temporal alignment & ConvLSTM tracking (Phase 4)
├── pinn/                   # Fisher-Kolmogorov reaction-diffusion PINN (Phase 4)
├── classifier/             # XGBoost multimodal recurrence classifier (Phase 5)
├── explainability/         # 3D Grad-CAM & SHAP attribution (Phase 6)
├── reporting/              # Slice renderers, volumetrics, & RANO reports (Phase 7)
├── training/               # Training engine, loss functions, & AMP
├── utils/                  # Structured logging & seed determinism
├── outputs/                # Checkpoints, logs, and generated reports
├── frontend/               # Full-Stack React Web Application
│   ├── index.html          # Web application shell
│   ├── app.jsx             # React 18 single-page dashboard application
│   └── styles.css          # Custom styling and animations
├── tests/                  # PyTest test suite
├── api.py                  # Full-Stack React Static Server & REST API
├── app.py                  # Interactive Streamlit Web Application
├── pipeline.py             # End-to-End 7-Phase Clinical Pipeline CLI
├── train.py                # Top-level 3D SegResNet Training CLI
├── validate.py             # Top-level Validation Metrics CLI
├── predict.py              # Top-level 3D Segmentation Inference CLI
└── visualize.py            # Multi-Modal MRI Slice Visualization CLI
```

---

## 🚀 How to Run (Step-by-Step Guide)

### 1. Prerequisites & Environment Setup

Ensure Python 3.10+ and Git are installed on your system.

```bash
# Clone the repository
git clone https://github.com/aaronshelley061/Physio-Rano.git
cd Physio-Rano

# Create and activate a virtual environment
# Windows (PowerShell):
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install all required dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 2. Dataset Ingestion & Caching

PhysioRANO supports automated downloading of the **BraTS 2024 GLI** cohort directly from Hugging Face:

```bash
# Automatically download and verify BraTS cohort
python -m datasets.downloader
```

*Data will be automatically cached under `data/brats2024_gli/` with four aligned MRI modalities (`t1n`, `t1c`, `t2w`, `t2f`) and ground truth segmentation masks (`seg.nii.gz`).*

---

### 3. Launching the Interactive Web Dashboards

PhysioRANO provides two distinct web dashboard interfaces depending on your preferred workflow:

#### Option A: Full-Stack React + REST API Dashboard (Recommended)
Launches the lightweight native Python REST API server and serves the responsive React 18 single-page interface:

```bash
python api.py
```
* **URL**: Open [http://localhost:8000](http://localhost:8000) in any modern browser.
* **Features**: Live multi-axial MRI slice exploration, interactive 3D SegResNet volume inference, PINN biophysical parameter tuning, 6-month cGAN longitudinal projection, SHAP explanation charts, and one-click PDF radiology report downloads.

#### Option B: Streamlit Research Dashboard
Launches the multi-tab Python analytical UI:

```bash
streamlit run app.py
```
* **URL**: Open [http://localhost:8501](http://localhost:8501) in your browser.
* **Features**: Dedicated research tabs for Preprocessing, 3D SegResNet Segmentation, PINN Biophysics, Recurrence Classification, Longitudinal Trajectory, Explainability (SHAP & Grad-CAM), and Clinical Reporting.

---

### 4. Running the Unified End-to-End Clinical Pipeline

Execute the full 7-phase clinical workflow on a specific patient subject from the command line:

```bash
# Run end-to-end pipeline on BraTS-GLI-00001-000
python pipeline.py --subject-id BraTS-GLI-00001-000

# Specify custom data or output directories
python pipeline.py --subject-id BraTS-GLI-00001-000 --data-dir data/brats2024_gli --output-dir outputs/reports
```

**Pipeline Output**:
* Ingests 4-modality 3D MRI scans and applies MONAI spatial transforms.
* Infers Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET) volumetric segmentations in $cm^3$.
* Solves Fisher-Kolmogorov reaction-diffusion PINN for diffusion ($D$) and proliferation ($\rho$) rates.
* Classifies True Progression vs. Pseudoprogression with calibrated probabilities.
* Evaluates RANO 2.0 response grade (CR, PR, SD, PD).
* Generates a downloadable clinical PDF summary report (`outputs/reports/<subject_id>_clinical_report.pdf`) and structured JSON report.

---

### 5. Model Training, Validation & Inference CLI

#### Train 3D SegResNet
Trains the 3D convolutional segmentation network using Hydra configuration, automatic mixed precision (AMP), and Dice Loss:
```bash
python train.py
```

#### Validate Model Checkpoints
Evaluates model weights (`best_model.pth`) on the validation split for Mean Dice and loss:
```bash
python validate.py
```

#### Run 3D Volume Prediction
Executes sliding-window inference on a subject volume and outputs a prediction NIfTI file:
```bash
python predict.py
```

#### Visualize Multi-Modal MRI Slices
Renders preprocessed axial slices for all four modalities alongside tumor sub-regions:
```bash
python visualize.py
```

#### Generate Clinical Benchmark Figures & Presentation Slides
Computes 5-fold cross-validation benchmarks, ROC curves, confusion matrices, and publication-ready 300 DPI figures:
```bash
python generate_presentation_results.py
```

---

### 6. Running the Automated Test Suite

Run the full automated PyTest test suite to verify RANO 2.0 logic, volume extraction, and pipeline components:

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=. -v
```

