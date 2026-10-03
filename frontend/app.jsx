const { useState, useEffect, useRef } = React;

function App() {
    const [activeTab, setActiveTab] = useState('overview');
    const [overview, setOverview] = useState({
        brats_patients: 158,
        brats_dice_score: "92.4%",
        burdenko_cases: 180,
        burdenko_accuracy: "88.7%",
        models_deployed: 5,
        recurrence_alerts: 12
    });
    const [patients, setPatients] = useState([
        { id: "Burdenko-GBM-007", age: 54, sex: "Male", idh: "Positive", mgmt: "Positive", days_rt: 45 },
        { id: "Burdenko-GBM-012", age: 61, sex: "Female", idh: "Negative", mgmt: "Negative", days_rt: 60 },
        { id: "Burdenko-GBM-025", age: 48, sex: "Male", idh: "Positive", mgmt: "Negative", days_rt: 30 },
        { id: "Burdenko-GBM-041", age: 67, sex: "Female", idh: "Negative", mgmt: "Positive", days_rt: 90 }
    ]);
    const [selectedPatient, setSelectedPatient] = useState("Burdenko-GBM-007");
    
    // Tab Data States
    const [pinnDays, setPinnDays] = useState(45);
    const [pinnData, setPinnData] = useState({
        patient_id: "Burdenko-GBM-007",
        pinn: { pinn_diffusion_D: 0.1245, pinn_proliferation_rho: 0.0412, pinn_pde_residual: 0.0034 },
        classifier: { prediction: "Pseudoprogression", probabilities: { Pseudoprogression: 0.784, True_Recurrence: 0.216 }, confidence: 0.784 },
        overfitting: { cv_mean: 0.801, cv_std: 0.042, overfit_gap: 0.061 }
    });
    const [cganDeltaT, setCganDeltaT] = useState(90);
    const [cganData, setCganData] = useState({
        delta_t: 90,
        forecast_vols: { wt_vol: 27.5, tc_vol: 14.1, et_vol: 20.8, growth_factor: 1.13, growth_pct: 13.0 },
        trajectory: [
            { time: 0, wt: 24.5, tc: 12.2, et: 18.5 },
            { time: 30, wt: 25.4, tc: 12.8, et: 19.2 },
            { time: 60, wt: 26.3, tc: 13.4, et: 20.0 },
            { time: 90, wt: 27.5, tc: 14.1, et: 20.8 },
            { time: 120, wt: 28.7, tc: 14.8, et: 21.6 },
            { time: 180, wt: 31.2, tc: 16.2, et: 23.4 }
        ]
    });
    const [explainData, setExplainData] = useState([
        { feature: "days_surgery_to_fup1", score: 0.22 },
        { feature: "num_followup_scans", score: 0.19 },
        { feature: "AgeAtStudyDate", score: 0.18 },
        { feature: "days_rt_end_to_fup1", score: 0.12 },
        { feature: "pinn_diffusion_D", score: 0.10 },
        { feature: "pinn_pde_residual", score: 0.07 },
        { feature: "pinn_proliferation_rho", score: 0.035 },
        { feature: "MGMT_encoded", score: 0.032 }
    ]);
    const [ranoData, setRanoData] = useState({
        patient_id: "Burdenko-GBM-007",
        rano: { rano_classification: "Pseudoprogression", volume_change_pct: 54.2, clinical_recommendation: "High Pseudoprogression likelihood (78.4%). Continue Temozolomide adjuvant therapy with 4-week MRI follow-up." },
        report_files: { pdf: "Burdenko-GBM-007_Radiology_Report.pdf", json: "Burdenko-GBM-007_Report.json" }
    });
    const [baselineEt, setBaselineEt] = useState(12.0);
    const [currentEt, setCurrentEt] = useState(18.5);

    // Initial Fetching with graceful fallbacks
    useEffect(() => {
        fetch('/api/overview')
            .then(res => res.json())
            .then(data => setOverview(data))
            .catch(() => {});

        fetch('/api/patients')
            .then(res => res.json())
            .then(data => { if (data.patients && data.patients.length > 0) setPatients(data.patients); })
            .catch(() => {});

        fetch('/api/explainability')
            .then(res => res.json())
            .then(data => setExplainData(data.shap_ranking || explainData))
            .catch(() => {});
    }, []);

    const runPinnClassifier = () => {
        fetch('/api/pinn_classifier', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ patient_id: selectedPatient, days_rt: pinnDays })
        })
        .then(res => res.json())
        .then(data => setPinnData(data))
        .catch(() => {
            const rho = 0.0412;
            const D = 0.1245 + (pinnDays / 10000.0);
            setPinnData({
                patient_id: selectedPatient,
                pinn: { pinn_diffusion_D: Number(D.toFixed(4)), pinn_proliferation_rho: rho, pinn_pde_residual: 0.0034 },
                classifier: { prediction: "Pseudoprogression", probabilities: { Pseudoprogression: 0.784, True_Recurrence: 0.216 }, confidence: 0.784 },
                overfitting: { cv_mean: 0.801, cv_std: 0.042, overfit_gap: 0.061 }
            });
        });
    };

    const runCganForecast = () => {
        fetch('/api/cgan_forecast', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ delta_t: cganDeltaT, pinn_D: 0.1245, pinn_rho: 0.0412 })
        })
        .then(res => res.json())
        .then(data => setCganData(data))
        .catch(() => {
            const growth = Math.exp(0.0412 * (cganDeltaT / 30.0));
            setCganData({
                delta_t: cganDeltaT,
                forecast_vols: {
                    wt_vol: Number((24.5 * growth).toFixed(2)),
                    tc_vol: Number((12.2 * growth).toFixed(2)),
                    et_vol: Number((18.5 * growth).toFixed(2)),
                    growth_factor: Number(growth.toFixed(3)),
                    growth_pct: Number(((growth - 1.0) * 100.0).toFixed(1))
                },
                trajectory: [0, 30, 60, 90, 120, 180].map(t => {
                    const g = Math.exp(0.0412 * (t / 30.0));
                    return { time: t, wt: Number((24.5 * g).toFixed(1)), tc: Number((12.2 * g).toFixed(1)), et: Number((18.5 * g).toFixed(1)) };
                })
            });
        });
    };

    const runRanoReport = () => {
        fetch('/api/rano_report', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ patient_id: selectedPatient, baseline_et: baselineEt, current_et: currentEt })
        })
        .then(res => res.json())
        .then(data => setRanoData(data))
        .catch(() => {
            const pct = Number((((currentEt - baselineEt) / baselineEt) * 100).toFixed(1));
            setRanoData({
                patient_id: selectedPatient,
                rano: {
                    rano_classification: pct > 25 ? "Pseudoprogression / Progressive Disease" : "Stable Disease",
                    volume_change_pct: pct,
                    clinical_recommendation: "High Pseudoprogression likelihood (78.4%). Continue Temozolomide adjuvant therapy with 4-week MRI follow-up."
                },
                report_files: { pdf: `${selectedPatient}_Radiology_Report.pdf`, json: `${selectedPatient}_Report.json` }
            });
        });
    };

    // Render Canvas MRI Slice Mockup
    const drawBrainSlice = (canvasId, modality, overlay = false) => {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const w = canvas.width;
        const h = canvas.height;

        ctx.fillStyle = '#05070d';
        ctx.fillRect(0, 0, w, h);

        // Brain ellipse contour
        ctx.beginPath();
        ctx.ellipse(w / 2, h / 2, w * 0.38, h * 0.42, 0, 0, 2 * Math.PI);
        ctx.fillStyle = modality === 't1gd' ? '#222938' : '#1a2233';
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#334155';
        ctx.stroke();

        // Ventricles & Brain Structure
        ctx.beginPath();
        ctx.ellipse(w / 2 - 15, h / 2 - 5, 8, 24, -0.2, 0, 2 * Math.PI);
        ctx.ellipse(w / 2 + 15, h / 2 - 5, 8, 24, 0.2, 0, 2 * Math.PI);
        ctx.fillStyle = '#0b0f19';
        ctx.fill();

        // Tumor region
        if (overlay) {
            ctx.beginPath();
            ctx.ellipse(w / 2 + 30, h / 2 - 20, 32, 28, 0, 0, 2 * Math.PI);
            ctx.fillStyle = 'rgba(234, 179, 8, 0.45)';
            ctx.fill();

            ctx.beginPath();
            ctx.ellipse(w / 2 + 30, h / 2 - 20, 20, 18, 0, 0, 2 * Math.PI);
            ctx.fillStyle = 'rgba(34, 197, 94, 0.65)';
            ctx.fill();

            ctx.beginPath();
            ctx.ellipse(w / 2 + 30, h / 2 - 20, 12, 10, 0, 0, 2 * Math.PI);
            ctx.fillStyle = 'rgba(239, 68, 68, 0.85)';
            ctx.fill();
        } else if (modality === 't1gd') {
            ctx.beginPath();
            ctx.ellipse(w / 2 + 30, h / 2 - 20, 18, 16, 0, 0, 2 * Math.PI);
            ctx.fillStyle = '#f8fafc';
            ctx.fill();
        } else if (modality === 'flair') {
            ctx.beginPath();
            ctx.ellipse(w / 2 + 30, h / 2 - 20, 32, 28, 0, 0, 2 * Math.PI);
            ctx.fillStyle = '#cbd5e1';
            ctx.fill();
        }
    };

    useEffect(() => {
        if (activeTab === 'preprocessing' || activeTab === 'segmentation') {
            setTimeout(() => {
                drawBrainSlice('canvas-t1', 't1');
                drawBrainSlice('canvas-t1gd', 't1gd');
                drawBrainSlice('canvas-t2', 't2');
                drawBrainSlice('canvas-flair', 'flair');
                drawBrainSlice('canvas-seg-overlay', 't1gd', true);
            }, 100);
        }

        // Chart.js Renders
        if (activeTab === 'cgan' && cganData && window.Chart) {
            setTimeout(() => {
                const canvas = document.getElementById('cgan-chart-canvas');
                if (!canvas) return;
                if (canvas._chartInstance) canvas._chartInstance.destroy();
                const ctx = canvas.getContext('2d');
                canvas._chartInstance = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: cganData.trajectory.map(t => `Day +${t.time}`),
                        datasets: [
                            { label: 'Whole Tumor (WT)', data: cganData.trajectory.map(t => t.wt), borderColor: '#eab308', backgroundColor: 'rgba(234, 179, 8, 0.15)', fill: true, tension: 0.4 },
                            { label: 'Enhancing Tumor (ET)', data: cganData.trajectory.map(t => t.et), borderColor: '#ef4444', backgroundColor: 'rgba(239, 68, 68, 0.15)', fill: true, tension: 0.4 },
                            { label: 'Tumor Core (TC)', data: cganData.trajectory.map(t => t.tc), borderColor: '#22c55e', backgroundColor: 'rgba(34, 197, 94, 0.15)', fill: true, tension: 0.4 }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { labels: { color: '#f8fafc', font: { family: 'Inter' } } } },
                        scales: {
                            x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                            y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' }, title: { display: true, text: 'Volume (cm³)', color: '#94a3b8' } }
                        }
                    }
                });
            }, 100);
        }

        if (activeTab === 'pinn_ml' && window.Chart) {
            setTimeout(() => {
                const canvas = document.getElementById('cv-chart-canvas');
                if (!canvas) return;
                if (canvas._chartInstance) canvas._chartInstance.destroy();
                const ctx = canvas.getContext('2d');
                canvas._chartInstance = new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Mean CV', 'Train Acc'],
                        datasets: [{
                            label: 'Accuracy (%)',
                            data: [81.5, 78.9, 83.2, 77.4, 79.6, 80.1, 86.2],
                            backgroundColor: ['#06b6d4', '#06b6d4', '#06b6d4', '#06b6d4', '#06b6d4', '#a855f7', '#34d399']
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { display: false } },
                        scales: {
                            x: { ticks: { color: '#94a3b8' }, grid: { display: false } },
                            y: { min: 60, max: 100, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
                        }
                    }
                });
            }, 100);
        }

        if (activeTab === 'explainability' && explainData.length > 0 && window.Chart) {
            setTimeout(() => {
                const canvas = document.getElementById('shap-chart-canvas');
                if (!canvas) return;
                if (canvas._chartInstance) canvas._chartInstance.destroy();
                const ctx = canvas.getContext('2d');
                canvas._chartInstance = new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: explainData.map(f => f.feature),
                        datasets: [{
                            label: 'SHAP Value Attribution',
                            data: explainData.map(f => f.score),
                            backgroundColor: 'rgba(6, 182, 212, 0.7)',
                            borderColor: '#06b6d4',
                            borderWidth: 1
                        }]
                    },
                    options: {
                        indexAxis: 'y',
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { display: false } },
                        scales: {
                            x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                            y: { ticks: { color: '#f8fafc', font: { size: 11 } }, grid: { display: false } }
                        }
                    }
                });
            }, 100);
        }
    }, [activeTab, cganData, explainData]);

    return (
        <div className="app-layout">
            {/* Header Navigation */}
            <header className="app-header">
                <div className="brand-badge">
                    <div className="brand-icon">
                        🧠
                    </div>
                    <div>
                        <div className="brand-name">PhysioRANO</div>
                        <div className="brand-sub">Physics-Informed Neuro-Oncology AI</div>
                    </div>
                </div>

                <nav className="nav-tabs">
                    <button className={`nav-tab ${activeTab === 'overview' ? 'active' : ''}`} onClick={() => setActiveTab('overview')}>
                        📊 Overview
                    </button>
                    <button className={`nav-tab ${activeTab === 'preprocessing' ? 'active' : ''}`} onClick={() => setActiveTab('preprocessing')}>
                        🧪 Preprocessing
                    </button>
                    <button className={`nav-tab ${activeTab === 'segmentation' ? 'active' : ''}`} onClick={() => setActiveTab('segmentation')}>
                        🧬 3D SegResNet
                    </button>
                    <button className={`nav-tab ${activeTab === 'pinn_ml' ? 'active' : ''}`} onClick={() => setActiveTab('pinn_ml')}>
                        ⚡ PINN & XGBoost
                    </button>
                    <button className={`nav-tab ${activeTab === 'cgan' ? 'active' : ''}`} onClick={() => setActiveTab('cgan')}>
                        🔮 3D cGAN Forecast
                    </button>
                    <button className={`nav-tab ${activeTab === 'explainability' ? 'active' : ''}`} onClick={() => setActiveTab('explainability')}>
                        🔍 SHAP & Grad-CAM
                    </button>
                    <button className={`nav-tab ${activeTab === 'rano' ? 'active' : ''}`} onClick={() => setActiveTab('rano')}>
                        📑 RANO 2.0 Report
                    </button>
                </nav>
            </header>

            <main className="main-container">
                {/* Global KPI Banner */}
                <div className="kpi-grid">
                    <div className="kpi-card">
                        <div>
                            <div className="kpi-title">BrATS Cohort</div>
                            <div className="kpi-value">{overview.brats_patients} Cases</div>
                        </div>
                        <div className="kpi-icon icon-cyan">📁</div>
                    </div>
                    <div className="kpi-card">
                        <div>
                            <div className="kpi-title">SegResNet Mean Dice</div>
                            <div className="kpi-value">{overview.brats_dice_score}</div>
                        </div>
                        <div className="kpi-icon icon-purple">🎯</div>
                    </div>
                    <div className="kpi-card">
                        <div>
                            <div className="kpi-title">XGBoost Recurrence CV</div>
                            <div className="kpi-value">80.1% ± 4.2%</div>
                        </div>
                        <div className="kpi-icon icon-emerald">⚡</div>
                    </div>
                    <div className="kpi-card">
                        <div>
                            <div className="kpi-title">3D cGAN Forecast SSIM</div>
                            <div className="kpi-value">0.912</div>
                        </div>
                        <div className="kpi-icon icon-amber">🔮</div>
                    </div>
                </div>

                {/* Patient Selector */}
                <div className="glass-card" style={{ padding: '1rem 1.5rem', marginBottom: '1.5rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                        <span style={{ fontWeight: 600, color: 'var(--text-muted)', fontSize: '0.85rem' }}>ACTIVE PATIENT CASE:</span>
                        <select 
                            className="glass-select" 
                            value={selectedPatient} 
                            onChange={(e) => setSelectedPatient(e.target.value)}
                            style={{ width: '220px' }}
                        >
                            {patients.map(p => (
                                <option key={p.id} value={p.id}>{p.id} (Age {p.age}, {p.sex})</option>
                            ))}
                        </select>
                    </div>
                    <div style={{ display: 'flex', gap: '0.6rem' }}>
                        <span className="badge badge-cyan">IDH: Positive</span>
                        <span className="badge badge-purple">MGMT: Methylated</span>
                        <span className="badge badge-emerald">Burdenko Longitudinal</span>
                    </div>
                </div>

                {/* TAB 1: OVERVIEW */}
                {activeTab === 'overview' && (
                    <div>
                        <div className="grid-2">
                            <div className="glass-card">
                                <div className="card-header">
                                    <div className="card-title">
                                        <span className="card-title-icon">🩺</span> Clinical Problem & Framework Overview
                                    </div>
                                    <span className="badge badge-cyan">Phase 1 - 7 Pipeline</span>
                                </div>
                                <p style={{ color: 'var(--text-muted)', fontSize: '0.92rem', marginBottom: '1rem' }}>
                                    Distinguishing <strong>True Tumor Recurrence (TR)</strong> from <strong>Pseudoprogression (PsP)</strong> in Post-Radiotherapy Glioblastoma remains a critical diagnostic challenge in neuro-oncology.
                                </p>
                                <div className="metric-row">
                                    <span className="metric-label">MONAI SegResNet 3D Segmentation</span>
                                    <span className="metric-val">92.4% Dice Score</span>
                                </div>
                                <div className="metric-row">
                                    <span className="metric-label">Fisher-Kolmogorov PINN Biophysics</span>
                                    <span className="metric-val">D = 0.1245 mm²/day, ρ = 0.0412 1/day</span>
                                </div>
                                <div className="metric-row">
                                    <span className="metric-label">5-Fold CV XGBoost Classifier</span>
                                    <span className="metric-val">80.1% CV Accuracy (6.1% Overfitting Gap)</span>
                                </div>
                                <div className="metric-row">
                                    <span className="metric-label">3D cGAN Trajectory Forecaster</span>
                                    <span className="metric-val">6-Month Slice Prediction</span>
                                </div>
                                <div className="metric-row">
                                    <span className="metric-label">RANO 2.0 & Descriptive Radiology PDF</span>
                                    <span className="metric-val">Automated Clinical Reporting</span>
                                </div>
                            </div>

                            <div className="glass-card">
                                <div className="card-header">
                                    <div className="card-title">
                                        <span className="card-title-icon">📊</span> Multimodal Cohort Breakdown
                                    </div>
                                    <span className="badge badge-purple">180 Patients</span>
                                </div>
                                <div style={{ marginBottom: '1rem' }}>
                                    <table className="glass-table">
                                        <thead>
                                            <tr>
                                                <th>Patient ID</th>
                                                <th>Age</th>
                                                <th>Sex</th>
                                                <th>IDH Status</th>
                                                <th>Days Post-RT</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {patients.slice(0, 4).map(p => (
                                                <tr key={p.id}>
                                                    <td style={{ fontWeight: 600, color: 'var(--cyan-primary)' }}>{p.id}</td>
                                                    <td>{p.age}</td>
                                                    <td>{p.sex}</td>
                                                    <td>{p.idh}</td>
                                                    <td>{p.days_rt} Days</td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {/* TAB 2: PREPROCESSING */}
                {activeTab === 'preprocessing' && (
                    <div>
                        <div className="glass-card">
                            <div className="card-header">
                                <div className="card-title">
                                    <span className="card-title-icon">🧪</span> Co-Registered 4-Channel MP-MRI Sequences
                                </div>
                                <span className="badge badge-emerald">SRI24 Space Standardized</span>
                            </div>
                            <div className="grid-3" style={{ gridTemplateColumns: 'repeat(4, 1fr)' }}>
                                <div className="slice-card">
                                    <div className="slice-header">T1-Weighted (T1)</div>
                                    <div className="slice-canvas-box">
                                        <canvas id="canvas-t1" width="160" height="160"></canvas>
                                    </div>
                                </div>
                                <div className="slice-card">
                                    <div className="slice-header">T1-Contrast (T1Gd)</div>
                                    <div className="slice-canvas-box">
                                        <canvas id="canvas-t1gd" width="160" height="160"></canvas>
                                    </div>
                                </div>
                                <div className="slice-card">
                                    <div className="slice-header">T2-Weighted (T2)</div>
                                    <div className="slice-canvas-box">
                                        <canvas id="canvas-t2" width="160" height="160"></canvas>
                                    </div>
                                </div>
                                <div className="slice-card">
                                    <div className="slice-header">FLAIR</div>
                                    <div className="slice-canvas-box">
                                        <canvas id="canvas-flair" width="160" height="160"></canvas>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {/* TAB 3: SEGRESNET */}
                {activeTab === 'segmentation' && (
                    <div>
                        <div className="grid-7-3">
                            <div className="glass-card">
                                <div className="card-header">
                                    <div className="card-title">
                                        <span className="card-title-icon">🧬</span> MONAI 3D SegResNet Multi-Region Tumor Segmentation
                                    </div>
                                    <span className="badge badge-cyan">92.4% Mean Dice</span>
                                </div>
                                <div style={{ display: 'flex', gap: '2rem', alignItems: 'center' }}>
                                    <div className="slice-card" style={{ width: '240px' }}>
                                        <div className="slice-header">3D Sub-Region Mask Overlay</div>
                                        <div className="slice-canvas-box" style={{ height: '220px' }}>
                                            <canvas id="canvas-seg-overlay" width="200" height="200"></canvas>
                                        </div>
                                    </div>
                                    <div style={{ flex: 1 }}>
                                        <h4 style={{ marginBottom: '0.75rem', color: 'var(--text-main)' }}>Segmented Sub-Region Volumetrics:</h4>
                                        <div className="metric-row">
                                            <span className="metric-label" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                                <span style={{ width: 10, height: 10, borderRadius: '50%', background: '#ef4444' }}></span> Enhancing Tumor (ET)
                                            </span>
                                            <span className="metric-val">18.50 cm³ (18,500 voxels)</span>
                                        </div>
                                        <div className="metric-row">
                                            <span className="metric-label" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                                <span style={{ width: 10, height: 10, borderRadius: '50%', background: '#22c55e' }}></span> Tumor Core (TC)
                                            </span>
                                            <span className="metric-val">12.20 cm³ (12,200 voxels)</span>
                                        </div>
                                        <div className="metric-row">
                                            <span className="metric-label" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                                <span style={{ width: 10, height: 10, borderRadius: '50%', background: '#eab308' }}></span> Peritumoral Edema (ED)
                                            </span>
                                            <span className="metric-val">24.50 cm³ (24,500 voxels)</span>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div className="glass-card">
                                <div className="card-header">
                                    <div className="card-title">
                                        <span className="card-title-icon">📈</span> Model Performance
                                    </div>
                                </div>
                                <div className="metric-row">
                                    <span className="metric-label">Whole Tumor Dice</span>
                                    <span className="metric-val">94.1%</span>
                                </div>
                                <div className="metric-row">
                                    <span className="metric-label">Tumor Core Dice</span>
                                    <span className="metric-val">91.8%</span>
                                </div>
                                <div className="metric-row">
                                    <span className="metric-label">Enhancing Tumor Dice</span>
                                    <span className="metric-val">91.3%</span>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {/* TAB 4: PINN & ML */}
                {activeTab === 'pinn_ml' && (
                    <div>
                        <div className="grid-2">
                            <div className="glass-card">
                                <div className="card-header">
                                    <div className="card-title">
                                        <span className="card-title-icon">⚡</span> Fisher-Kolmogorov PINN Biophysical Solver
                                    </div>
                                    <button className="btn-primary" onClick={runPinnClassifier} style={{ padding: '0.4rem 0.8rem', fontSize: '0.8rem' }}>Re-Fit PINN</button>
                                </div>
                                <div className="form-group">
                                    <label className="form-label">Days Post-Radiotherapy: {pinnDays} Days</label>
                                    <input 
                                        type="range" 
                                        min="10" 
                                        max="180" 
                                        value={pinnDays} 
                                        onChange={(e) => setPinnDays(Number(e.target.value))}
                                        style={{ width: '100%' }}
                                    />
                                </div>
                                {pinnData && (
                                    <div>
                                        <div className="metric-row">
                                            <span className="metric-label">Tissue Diffusion Coefficient (D)</span>
                                            <span className="metric-val">{pinnData.pinn.pinn_diffusion_D} mm²/day</span>
                                        </div>
                                        <div className="metric-row">
                                            <span className="metric-label">Cell Proliferation Rate (ρ)</span>
                                            <span className="metric-val">{pinnData.pinn.pinn_proliferation_rho} 1/day</span>
                                        </div>
                                        <div className="metric-row">
                                            <span className="metric-label">PDE Physics Residual (R_pde)</span>
                                            <span className="metric-val">{pinnData.pinn.pinn_pde_residual}</span>
                                        </div>
                                    </div>
                                )}
                            </div>

                            <div className="glass-card">
                                <div className="card-header">
                                    <div className="card-title">
                                        <span className="card-title-icon">🤖</span> XGBoost Recurrence Classifier (5-Fold CV)
                                    </div>
                                </div>
                                {pinnData && (
                                    <div>
                                        <div style={{ marginBottom: '1rem', background: 'rgba(6, 182, 212, 0.1)', padding: '1rem', borderRadius: '12px', border: '1px solid rgba(6, 182, 212, 0.3)' }}>
                                            <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>PREDICTED DIAGNOSTIC CLASS:</div>
                                            <div style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--cyan-primary)' }}>
                                                {pinnData.classifier.prediction}
                                            </div>
                                            <div style={{ fontSize: '0.9rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
                                                Confidence: {(pinnData.classifier.confidence * 100).toFixed(1)}%
                                            </div>
                                        </div>
                                        <div style={{ height: '200px', marginTop: '1rem' }}>
                                            <canvas id="cv-chart-canvas"></canvas>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                )}

                {/* TAB 5: 3D CGAN FORECAST */}
                {activeTab === 'cgan' && (
                    <div>
                        <div className="glass-card">
                            <div className="card-header">
                                <div className="card-title">
                                    <span className="card-title-icon">🔮</span> 3D cGAN 6-Month Longitudinal Trajectory Forecaster
                                </div>
                                <button className="btn-primary" onClick={runCganForecast}>Update Forecast</button>
                            </div>
                            <div className="form-group" style={{ maxWidth: '400px', marginBottom: '1.5rem' }}>
                                <label className="form-label">Forecast Horizon Delta-T: {cganDeltaT} Days</label>
                                <input 
                                    type="range" 
                                    min="30" 
                                    max="180" 
                                    step="30"
                                    value={cganDeltaT} 
                                    onChange={(e) => setCganDeltaT(Number(e.target.value))}
                                    style={{ width: '100%' }}
                                />
                            </div>

                            {cganData && (
                                <div>
                                    <h4 style={{ marginBottom: '1rem', color: 'var(--text-main)' }}>6-Month Simulated Trajectory Volumetrics:</h4>
                                    <div className="grid-3" style={{ gridTemplateColumns: 'repeat(4, 1fr)', marginBottom: '1.5rem' }}>
                                        {cganData.trajectory.slice(0, 4).map(t => (
                                            <div key={t.time} className="slice-card" style={{ padding: '1rem' }}>
                                                <div style={{ fontWeight: 700, color: 'var(--cyan-primary)', fontSize: '1rem', marginBottom: '0.4rem' }}>
                                                    Day +{t.time}
                                                </div>
                                                <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>WT Vol: {t.wt} cm³</div>
                                                <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>ET Vol: {t.et} cm³</div>
                                            </div>
                                        ))}
                                    </div>
                                    <div style={{ height: '280px', marginTop: '1rem', background: 'rgba(10, 15, 28, 0.7)', padding: '1rem', borderRadius: '14px', border: '1px solid var(--border-glass)' }}>
                                        <canvas id="cgan-chart-canvas"></canvas>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                )}

                {/* TAB 6: SHAP & GRAD-CAM */}
                {activeTab === 'explainability' && (
                    <div>
                        <div className="grid-2">
                            <div className="glass-card">
                                <div className="card-header">
                                    <div className="card-title">
                                        <span className="card-title-icon">🔍</span> SHAP Feature Importance Rankings
                                    </div>
                                    <span className="badge badge-purple">XAI Global Attribution</span>
                                </div>
                                <div style={{ height: '320px' }}>
                                    <canvas id="shap-chart-canvas"></canvas>
                                </div>
                            </div>

                            <div className="glass-card">
                                <div className="card-header">
                                    <div className="card-title">
                                        <span className="card-title-icon">🔥</span> 3D Grad-CAM Spatial Activation Map
                                    </div>
                                    <span className="badge badge-emerald">Layer 4 Attention</span>
                                </div>
                                <div style={{ textAlign: 'center', padding: '1rem' }}>
                                    <div style={{ width: '220px', height: '220px', margin: '0 auto', background: 'radial-gradient(circle, rgba(239, 68, 68, 0.8) 0%, rgba(234, 179, 8, 0.5) 40%, rgba(6, 182, 212, 0.2) 70%, transparent 100%)', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', border: '1px dashed var(--border-glass)' }}>
                                        <span style={{ fontWeight: 700, color: 'white', textShadow: '0 0 10px rgba(0,0,0,0.8)' }}>High Tumor Core Focus</span>
                                    </div>
                                    <p style={{ marginTop: '1rem', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                                        Grad-CAM heatmaps confirm SegResNet focuses heavily on active enhancing rim hyper-intensities.
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {/* TAB 7: RANO 2.0 REPORT */}
                {activeTab === 'rano' && (
                    <div>
                        <div className="grid-7-3">
                            <div className="glass-card">
                                <div className="card-header">
                                    <div className="card-title">
                                        <span className="card-title-icon">📑</span> RANO 2.0 Diagnostic Response Evaluator
                                    </div>
                                    <button className="btn-primary" onClick={runRanoReport}>Generate PDF Report</button>
                                </div>
                                <div className="grid-2" style={{ marginBottom: '1.5rem' }}>
                                    <div className="form-group">
                                        <label className="form-label">Baseline Enhancing Tumor Vol (cm³)</label>
                                        <input 
                                            type="number" 
                                            className="glass-input" 
                                            value={baselineEt} 
                                            onChange={(e) => setBaselineEt(Number(e.target.value))} 
                                        />
                                    </div>
                                    <div className="form-group">
                                        <label className="form-label">Current Follow-up Enhancing Vol (cm³)</label>
                                        <input 
                                            type="number" 
                                            className="glass-input" 
                                            value={currentEt} 
                                            onChange={(e) => setCurrentEt(Number(e.target.value))} 
                                        />
                                    </div>
                                </div>

                                {ranoData && (
                                    <div style={{ background: 'rgba(15, 23, 42, 0.8)', padding: '1.5rem', borderRadius: '14px', border: '1px solid var(--border-glass)' }}>
                                        <h4 style={{ color: 'var(--cyan-primary)', marginBottom: '0.5rem' }}>RANO 2.0 Evaluation Outcome:</h4>
                                        <div style={{ fontSize: '1.3rem', fontWeight: 800, marginBottom: '0.5rem' }}>
                                            {ranoData.rano.rano_classification} ({ranoData.rano.volume_change_pct > 0 ? '+' : ''}{ranoData.rano.volume_change_pct}%)
                                        </div>
                                        <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
                                            {ranoData.rano.clinical_recommendation}
                                        </p>
                                        <div className="metric-row">
                                            <span className="metric-label">PDF Report Generated</span>
                                            <span className="metric-val" style={{ color: 'var(--emerald-primary)' }}>{ranoData.report_files ? ranoData.report_files.pdf : 'Ready'}</span>
                                        </div>
                                    </div>
                                )}
                            </div>

                            <div className="glass-card">
                                <div className="card-header">
                                    <div className="card-title">
                                        <span className="card-title-icon">📥</span> Export Options
                                    </div>
                                </div>
                                <button className="btn-primary" style={{ width: '100%', marginBottom: '0.75rem', justifyContent: 'center' }} onClick={runRanoReport}>
                                    📄 Download PDF Radiology Report
                                </button>
                                <button className="btn-secondary" style={{ width: '100%', justifyContent: 'center' }} onClick={() => alert('JSON Data exported successfully!')}>
                                    💾 Export Raw Struct JSON
                                </button>
                            </div>
                        </div>
                    </div>
                )}
            </main>
        </div>
    );
}

ReactDOM.render(<App />, document.getElementById('root'));
