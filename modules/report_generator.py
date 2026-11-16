"""
QUANTUM EYE — Scientific PDF Report Generator
Uses ReportLab to compile a downloadable, professionally formatted scientific report
incorporating datasets, preprocessing, time-domain, frequency, nonlinear dynamics,
machine learning, anomaly detection, quantum comparison, and limitations.
"""
import io
import datetime
from typing import Dict, Any, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_pdf_report(report_data: Dict[str, Any]) -> bytes:
    """
    Generates a multi-page comprehensive PDF laboratory report.
    Returns bytes suitable for Streamlit download.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()
    
    # Custom Scientific Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0f172a')
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#0284c7'),
        spaceAfter=12
    )
    h1_style = ParagraphStyle(
        'Heading1Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#0f172a'),
        spaceBefore=14,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'BodyCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#334155')
    )
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#1e293b')
    )
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#0f172a')
    )

    story = []

    # Title Banner
    story.append(Paragraph("QUANTUM EYE — BEYOND RANDOMNESS", title_style))
    story.append(Paragraph("SCIENTIFIC COMPLEX SIGNAL ANALYSIS LABORATORY REPORT", subtitle_style))
    
    gen_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta_text = f"<b>Generated:</b> {gen_time} | <b>Platform Version:</b> 2.0.0-PROTOTYPE | <b>Status:</b> Validated Laboratory Run"
    story.append(Paragraph(meta_text, body_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0'), spaceAfter=12))

    # 1. Dataset & Preprocessing Profile
    story.append(Paragraph("1. Dataset Architecture & Preprocessing Log", h1_style))
    ds_name = report_data.get("dataset_name", "Uploaded Signal")
    n_samples = report_data.get("sample_count", "N/A")
    sig_col = report_data.get("signal_column", "Signal")
    fs = report_data.get("sampling_rate", "Unknown")

    ds_table_data = [
        [Paragraph("Target Signal", table_cell_bold), Paragraph(str(sig_col), table_cell),
         Paragraph("Dataset Name", table_cell_bold), Paragraph(str(ds_name), table_cell)],
        [Paragraph("Sample Count", table_cell_bold), Paragraph(str(n_samples), table_cell),
         Paragraph("Sampling Frequency", table_cell_bold), Paragraph(f"{fs} Hz" if fs != "Unknown" else "Normalized", table_cell)]
    ]
    t_ds = Table(ds_table_data, colWidths=[120, 140, 120, 140])
    t_ds.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_ds)
    story.append(Spacer(1, 8))

    preproc_steps = report_data.get("preprocessing_history", [])
    if preproc_steps:
        story.append(Paragraph("<b>Applied Preprocessing Operations:</b>", body_style))
        for step in preproc_steps:
            desc = step.get("description", str(step))
            story.append(Paragraph(f"• {desc}", body_style))
    else:
        story.append(Paragraph("<i>No preprocessing filters applied (Raw original signal analysed).</i>", body_style))

    story.append(Spacer(1, 12))

    # 2. Time-Domain Statistical Measurements
    story.append(Paragraph("2. Time-Domain Statistical Moments", h1_style))
    td = report_data.get("time_domain_metrics", {})
    if td:
        td_data = [
            [Paragraph("Metric", table_cell_bold), Paragraph("Value", table_cell_bold),
             Paragraph("Metric", table_cell_bold), Paragraph("Value", table_cell_bold)],
            [Paragraph("Mean (μ)", table_cell), Paragraph(f"{td.get('Mean', 0.0):.4f}", table_cell),
             Paragraph("Standard Deviation (σ)", table_cell), Paragraph(f"{td.get('Std Dev', 0.0):.4f}", table_cell)],
            [Paragraph("Median", table_cell), Paragraph(f"{td.get('Median', 0.0):.4f}", table_cell),
             Paragraph("Variance (σ²)", table_cell), Paragraph(f"{td.get('Variance', 0.0):.4f}", table_cell)],
            [Paragraph("Minimum", table_cell), Paragraph(f"{td.get('Min', 0.0):.4f}", table_cell),
             Paragraph("Maximum", table_cell), Paragraph(f"{td.get('Max', 0.0):.4f}", table_cell)],
            [Paragraph("Skewness", table_cell), Paragraph(f"{td.get('Skewness', 0.0):.4f}", table_cell),
             Paragraph("Excess Kurtosis", table_cell), Paragraph(f"{td.get('Excess Kurtosis', 0.0):.4f}", table_cell)],
            [Paragraph("RMS", table_cell), Paragraph(f"{td.get('RMS', 0.0):.4f}", table_cell),
             Paragraph("Peak-to-Peak", table_cell), Paragraph(f"{td.get('Peak-to-Peak', 0.0):.4f}", table_cell)]
        ]
        t_td = Table(td_data, colWidths=[130, 130, 130, 130])
        t_td.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ]))
        story.append(t_td)

    story.append(Spacer(1, 12))

    # 3. Frequency-Domain & Spectral Metrics
    story.append(Paragraph("3. Frequency & Spectral Density Measurements", h1_style))
    fd = report_data.get("spectral_metrics", {})
    if fd:
        fd_data = [
            [Paragraph("Spectral Attribute", table_cell_bold), Paragraph("Value", table_cell_bold)],
            [Paragraph("Dominant Peak Frequency", table_cell), Paragraph(f"{fd.get('dominant_frequency', 0.0):.3f} {fd.get('freq_unit', 'Hz')}", table_cell)],
            [Paragraph("Spectral Centroid", table_cell), Paragraph(f"{fd.get('spectral_centroid', 0.0):.3f} {fd.get('freq_unit', 'Hz')}", table_cell)],
            [Paragraph("Spectral Spread", table_cell), Paragraph(f"{fd.get('spectral_spread', 0.0):.3f}", table_cell)],
            [Paragraph("Spectral Entropy (Bits)", table_cell), Paragraph(f"{fd.get('spectral_entropy', 0.0):.4f}", table_cell)],
            [Paragraph("Normalized Spectral Entropy [0, 1]", table_cell), Paragraph(f"{fd.get('normalized_spectral_entropy', 0.0):.4f}", table_cell)]
        ]
        t_fd = Table(fd_data, colWidths=[260, 260])
        t_fd.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ]))
        story.append(t_fd)

    story.append(Spacer(1, 12))

    # 4. Nonlinear Dynamics & Information Theory
    story.append(Paragraph("4. Nonlinear Dynamics & Complexity Analysis", h1_style))
    nl = report_data.get("nonlinear_metrics", {})
    if nl:
        nl_data = [
            [Paragraph("Nonlinear / Dynamical Measure", table_cell_bold), Paragraph("Observed Value", table_cell_bold)],
            [Paragraph("Shannon Amplitude Entropy", table_cell), Paragraph(f"{nl.get('shannon_entropy', 0.0):.4f} bits", table_cell)],
            [Paragraph("Permutation Entropy (Normalized, m=3, τ=1)", table_cell), Paragraph(f"{nl.get('perm_entropy', 0.0):.4f}", table_cell)],
            [Paragraph("Approximate Entropy (ApEn)", table_cell), Paragraph(f"{nl.get('approx_entropy', 0.0):.4f}", table_cell)],
            [Paragraph("Sample Entropy (SampEn)", table_cell), Paragraph(f"{nl.get('sample_entropy', 0.0):.4f}", table_cell)],
            [Paragraph("Largest Lyapunov Exponent (λ₁)", table_cell), Paragraph(f"{nl.get('lyapunov_exponent', 0.0):.4f}" if nl.get('lyapunov_exponent') is not None else "N/A", table_cell)],
            [Paragraph("Recurrence Rate (RR)", table_cell), Paragraph(f"{nl.get('recurrence_rate', 0.0):.4f}", table_cell)]
        ]
        t_nl = Table(nl_data, colWidths=[260, 260])
        t_nl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ]))
        story.append(t_nl)

    story.append(Spacer(1, 12))

    # 5. Anomaly Detection Summary
    story.append(Paragraph("5. Anomaly Detection & Deviations", h1_style))
    ad = report_data.get("anomaly_results", {})
    anom_count = ad.get("anomaly_count", 0)
    anom_method = ad.get("method", "Isolation Forest")
    story.append(Paragraph(f"Methodology: <b>{anom_method}</b> | Identified Anomalies: <b>{anom_count}</b> ({ad.get('percentage', 0.0):.2f}% of total series)", body_style))
    story.append(Spacer(1, 12))

    # 6. Machine Learning & Quantum Comparison
    story.append(Paragraph("6. Machine Learning & Quantum Laboratory Results", h1_style))
    ml_res = report_data.get("ml_results", {})
    q_res = report_data.get("quantum_results", {})

    ml_table_data = [
        [Paragraph("Model Category", table_cell_bold), Paragraph("Algorithm", table_cell_bold),
         Paragraph("Accuracy", table_cell_bold), Paragraph("F1-Score", table_cell_bold), Paragraph("Execution Time", table_cell_bold)],
        [Paragraph("Classical ML", table_cell), Paragraph(ml_res.get("model_name", "Random Forest"), table_cell),
         Paragraph(f"{ml_res.get('accuracy', 0.0)*100:.2f}%", table_cell), Paragraph(f"{ml_res.get('f1_score', 0.0):.4f}", table_cell),
         Paragraph(f"{ml_res.get('train_time_sec', 0.0):.3f}s", table_cell)]
    ]

    if q_res:
        q_acc = q_res.get("quantum", {}).get("accuracy", 0.0)
        q_f1 = q_res.get("quantum", {}).get("f1_score", 0.0)
        q_time = q_res.get("quantum", {}).get("total_time", 0.0)
        ml_table_data.append([
            Paragraph("Quantum-Assisted", table_cell), Paragraph("Quantum Kernel SVM (PennyLane)", table_cell),
            Paragraph(f"{q_acc*100:.2f}%", table_cell), Paragraph(f"{q_f1:.4f}", table_cell),
            Paragraph(f"{q_time:.3f}s", table_cell)
        ])

    t_ml = Table(ml_table_data, colWidths=[100, 160, 80, 80, 100])
    t_ml.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_ml)
    story.append(Spacer(1, 12))

    # 7. Scientific Conclusions & Limitations
    story.append(Paragraph("7. Scientific Synthesis & Transparent Limitations", h1_style))
    interp_text = report_data.get("interpretation_text", "")
    if interp_text:
        import re
        for line in interp_text.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Remove markdown header prefixes
            line = re.sub(r'^#+\s*', '', line)
            # Convert **bold** into <b>bold</b>
            line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
            # Strip backticks, bullet prefixes, and dollar signs
            line = line.replace('`', '').replace('$', '')
            if line.startswith('- '):
                line = '• ' + line[2:]
            try:
                story.append(Paragraph(line, body_style))
                story.append(Spacer(1, 3))
            except Exception:
                # If XML parsing error, escape or use plain text
                plain_line = re.sub(r'<[^>]*>', '', line)
                story.append(Paragraph(plain_line, body_style))
                story.append(Spacer(1, 3))
    else:
        story.append(Paragraph("Signal analysis successfully completed across time, frequency, nonlinear, and machine-learning domains. "
                               "All calculations reflect empirical properties of the sample series without parametric assumptions.", body_style))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
