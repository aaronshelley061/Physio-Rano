"""
PhysioRANO Medical Image Segmentation Models Comparison PDF Exporter.

Generates a publication-quality, landscape PDF report comparing
SegResNet against established 3D medical segmentation models.
"""

from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_comparison_pdf(output_path: str = "outputs/reports/medical_segmentation_comparison.pdf") -> Path:
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 11 x 8.5 inches landscape (792 x 612 pt), 36pt (0.5 inch) margins -> 720 pt width
    doc = SimpleDocTemplate(
        str(out_file),
        pagesize=landscape(letter),
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0F2942"),
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#4A5568"),
    )

    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#1A365D"),
        spaceBefore=12,
        spaceAfter=6,
    )

    th_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        alignment=0,
    )

    td_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#2D3748"),
    )

    td_bold = ParagraphStyle(
        "TableCellBold",
        parent=td_style,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#0F2942"),
    )

    td_green = ParagraphStyle(
        "TableCellGreen",
        parent=td_style,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#1B6440"),
    )

    td_red = ParagraphStyle(
        "TableCellRed",
        parent=td_style,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#9B2C2C"),
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#2D3748"),
    )

    bullet_style = ParagraphStyle(
        "BulletDark",
        parent=body_style,
        leftIndent=12,
        spaceAfter=3,
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("PhysioRANO &mdash; 3D Medical Segmentation Architecture Evaluation", title_style))
    story.append(Paragraph(
        f"<b>Comparative Analysis:</b> Established 3D CNN Architectures for Multi-Parametric Brain MRI (BraTS) &nbsp;|&nbsp; "
        f"<b>Generated:</b> {datetime.now().strftime('%B %d, %Y')} &nbsp;|&nbsp; <b>Artifact:</b> Architectural Justification Document",
        subtitle_style
    ))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3182CE"), spaceAfter=10))

    # Table columns definition: total width = 720 pt
    # Col widths: Model(90), Arch(85), Params(60), VRAM(70), Batch(75), GradCAM(75), Strengths(130), Limitations(135)
    col_widths = [90, 85, 60, 70, 75, 75, 130, 135]

    headers = [
        Paragraph("Model", th_style),
        Paragraph("Architecture Type", th_style),
        Paragraph("Params", th_style),
        Paragraph("VRAM (3D)", th_style),
        Paragraph("Batch Stab. (B≤2)", th_style),
        Paragraph("3D Grad-CAM", th_style),
        Paragraph("Key Strengths", th_style),
        Paragraph("Limitations", th_style),
    ]

    data = [headers]

    rows_def = [
        (
            Paragraph("<b>SegResNet</b><br/><font color='#1B6440'><b>[Selected]</b></font><br/>(Myronenko 2018)", td_bold),
            Paragraph("Asymmetric Residual 3D CNN", td_style),
            Paragraph("<b>~4.7 M</b>", td_green),
            Paragraph("<b>Low (~4–6 GB)</b>", td_green),
            Paragraph("<b>High</b><br/>(GroupNorm)", td_green),
            Paragraph("<b>Native & Direct</b><br/>(clean conv maps)", td_green),
            Paragraph("• Specifically designed & proven on BraTS<br/>• Compact decoder speeds up inference<br/>• Smooth CPU / consumer GPU inference", td_style),
            Paragraph("• Slightly less generic than automated hyper-parameter tuning frameworks like nnU-Net", td_style),
        ),
        (
            Paragraph("<b>Standard 3D U-Net</b><br/>(Çiçek et al. 2016)", td_bold),
            Paragraph("Symmetric Encoder-Decoder 3D CNN", td_style),
            Paragraph("~16–30 M", td_style),
            Paragraph("High (~10–14 GB)", td_style),
            Paragraph("<b>Low</b><br/>(BatchNorm fails at B=1)", td_red),
            Paragraph("Native", td_style),
            Paragraph("• Foundational 3D medical baseline<br/>• Intuitive symmetric skip connections", td_style),
            Paragraph("• Heavy symmetric decoder<br/>• High VRAM consumption<br/>• Slow sliding-window inference", td_style),
        ),
        (
            Paragraph("<b>nnU-Net</b><br/>(Isensee et al. 2021)", td_bold),
            Paragraph("Self-Configuring 3D U-Net Framework", td_style),
            Paragraph("~20–30 M", td_style),
            Paragraph("High (~12–16 GB)", td_style),
            Paragraph("<b>High</b><br/>(InstanceNorm)", td_green),
            Paragraph("Native", td_style),
            Paragraph("• Out-of-the-box state-of-the-art across 50+ biomedical challenges<br/>• Automated fingerprint-based heuristics", td_style),
            Paragraph("• Heavy pipeline dependencies<br/>• High compute & memory footprint<br/>• Cumbersome for lightweight real-time dashboards", td_style),
        ),
        (
            Paragraph("<b>Attention U-Net</b><br/>(Oktay et al. 2018)", td_bold),
            Paragraph("3D U-Net + Additive Attention Gates", td_style),
            Paragraph("~25–35 M", td_style),
            Paragraph("High (~12–15 GB)", td_style),
            Paragraph("Medium / Low<br/>(norm-dependent)", td_style),
            Paragraph("Native", td_style),
            Paragraph("• Suppresses background tissue noise<br/>• Dynamically weights salient lesion regions", td_style),
            Paragraph("• Computationally demanding attention gating<br/>• Noticeably slower training & inference convergence", td_style),
        ),
        (
            Paragraph("<b>V-Net</b><br/>(Milletari et al. 2016)", td_bold),
            Paragraph("Residual 3D CNN with Soft Dice Loss", td_style),
            Paragraph("~45 M", td_style),
            Paragraph("<b>Very High (&gt;14 GB)</b>", td_red),
            Paragraph("Medium", td_style),
            Paragraph("Native", td_style),
            Paragraph("• Pioneer of volumetric 3D loss & residual training for organ segmentation", td_style),
            Paragraph("• Obsolete wide filter counts<br/>• Excessive parameter redundancy<br/>• High risk of CUDA OOM on non-workstation GPUs", td_style),
        ),
        (
            Paragraph("<b>HighRes3DNet</b><br/>(Li et al. 2017)", td_bold),
            Paragraph("Dilated Residual 3D CNN (No Downsampling)", td_style),
            Paragraph("~3–5 M", td_style),
            Paragraph("<b>Very High (&gt;16 GB)</b>", td_red),
            Paragraph("High", td_green),
            Paragraph("Native", td_style),
            Paragraph("• Retains full spatial voxel resolution without pooling<br/>• Superb micro-boundary delineation", td_style),
            Paragraph("• Enormous intermediate tensor activations<br/>• High memory consumption despite low parameter count", td_style),
        ),
    ]

    for row in rows_def:
        data.append(row)

    comp_table = Table(data, colWidths=col_widths, repeatRows=1)
    comp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1A365D")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        # Highlight SegResNet row
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor("#EBF8FF")),
        # Alternating rows
        ('BACKGROUND', (0, 2), (-1, 2), colors.white),
        ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor("#F7FAFC")),
        ('BACKGROUND', (0, 4), (-1, 4), colors.white),
        ('BACKGROUND', (0, 5), (-1, 5), colors.HexColor("#F7FAFC")),
        ('BACKGROUND', (0, 6), (-1, 6), colors.white),
        # Grid lines
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#718096")),
        ('LINEBELOW', (0, 1), (-1, 1), 1.2, colors.HexColor("#3182CE")),
    ]))

    story.append(comp_table)
    story.append(Spacer(1, 10))

    # Justification Section
    justification_elements = [
        Paragraph("Key Rationale: Why SegResNet Was Chosen for the PhysioRANO Pipeline", h2_style),
        Paragraph(
            "<b>1. Domain Optimization for BraTS Sub-Regions:</b> "
            "SegResNet was custom-crafted to delineate multi-parametric MRI modalities (T1, T1c, T2, FLAIR) "
            "into hierarchical overlapping targets: Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET). "
            "It holds consistent top-tier rank on international benchmark competitions.",
            bullet_style
        ),
        Paragraph(
            "<b>2. Extreme Parameter & VRAM Efficiency (~4.7M vs ~25–45M):</b> "
            "By utilizing an asymmetric encoder-decoder configuration (deep encoder blocks [1,2,2,4] paired with single-pass upsampling [1,1,1]), "
            "SegResNet dramatically reduces memory bandwidth. It delivers near-lossless Dice accuracy while consuming less than half the VRAM of standard 3D U-Net or V-Net.",
            bullet_style
        ),
        Paragraph(
            "<b>3. Small Batch Size Robustness via Group Normalization:</b> "
            "3D medical volumes mandate small training batches (B=1 or 2 per GPU). Conventional Batch Normalization (used in baseline 3D U-Net) suffers catastrophic variance degradation at B≤2. "
            "SegResNet computes statistics across channel groups per volume, providing rock-solid convergence.",
            bullet_style
        ),
        Paragraph(
            "<b>4. Interactive Web Dashboard & Real-Time Clinical Inference:</b> "
            "PhysioRANO embeds inference directly into an interactive Streamlit clinical workstation. "
            "SegResNet performs sliding-window 3D inference in 3–5 seconds on standard GPUs and CPU, preventing timeouts and CUDA OOM failures.",
            bullet_style
        ),
        Paragraph(
            "<b>5. Native Explainability & Biophysical Coupling:</b> "
            "The model provides clean convolutional feature activations for 3D Grad-CAM spatial heatmaps, "
            "while its volume estimates seamlessly initialize the Fisher-Kolmogorov biophysical reaction-diffusion PINN and RANO 2.0 response engine.",
            bullet_style
        ),
    ]

    story.append(KeepTogether(justification_elements))

    doc.build(story)
    return out_file

if __name__ == "__main__":
    out = generate_comparison_pdf()
    print(f"Comparison PDF successfully generated at: {out.resolve()}")
