"""
Report Generator Module for Customer Segmentation.
Generates academic PDF project reports and exports clustered dataset summaries.
"""

import os
import io
import pandas as pd
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Adds page numbers and footer to all pages."""
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#64748b"))
        
        # Header line & title
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(54, 750, 558, 750)
        self.drawString(54, 755, "Customer Segmentation via K-Means Clustering — Academic Project Report")
        
        # Footer
        self.line(54, 45, 558, 45)
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 32, page_str)
        self.drawString(54, 32, f"Confidential & Academic ML Demo | Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        self.restoreState()


def generate_pdf_report(dataset_summary: dict, model_bundle: dict, output_path: str = None) -> bytes:
    """
    Generate an academic-grade PDF project report using ReportLab.
    Returns bytes or writes to output_path.
    """
    buffer = io.BytesIO() if output_path is None else open(output_path, 'wb')
    doc = SimpleDocTemplate(
        buffer if output_path is None else output_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#475569'),
        spaceAfter=15
    )

    heading2_style = ParagraphStyle(
        'Heading2Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#1e3a8a'),
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'BodyCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#334155'),
        spaceAfter=8
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor('#1e293b')
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("ACADEMIC PROJECT REPORT", ParagraphStyle('Tag', fontName='Helvetica-Bold', fontSize=10, textColor=colors.HexColor('#2563eb'))))
    story.append(Paragraph("Customer Purchasing Behaviour Segmentation Using K-Means Clustering", title_style))
    story.append(Paragraph("A machine learning research and implementation study evaluating Mall Customer purchasing patterns using K-Means++ clustering, Silhouette analysis, and dynamic persona profiling.", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2563eb'), spaceBefore=2, spaceAfter=12))

    # Executive Summary
    story.append(Paragraph("1. Executive Summary & Problem Formulation", heading2_style))
    summary_text = (
        "Customer segmentation is an essential unsupervised learning strategy that enables retailers to partition "
        "heterogeneous consumer bases into distinct, homogeneous subgroups based on behavioral and financial attributes. "
        "This project implements an end-to-end K-Means clustering pipeline on the Mall Customer dataset, leveraging "
        "StandardScaler for feature standardization, the Elbow Method (WCSS) and Silhouette Score evaluation to select an optimal K, "
        "and multi-dimensional profiling to derive actionable marketing recommendations."
    )
    story.append(Paragraph(summary_text, body_style))

    # Dataset Summary Table
    story.append(Paragraph("2. Dataset Exploration & Characteristics", heading2_style))
    total_records = dataset_summary.get('total_records', 200)
    kpis = dataset_summary.get('kpi', {})
    avg_inc = kpis.get('avg_income', 'N/A')
    avg_sp = kpis.get('avg_spending', 'N/A')
    avg_age = kpis.get('avg_age', 'N/A')

    dataset_table_data = [
        [Paragraph("Metric", table_header_style), Paragraph("Value", table_header_style), Paragraph("Description", table_header_style)],
        [Paragraph("Total Customer Records", table_cell_style), Paragraph(str(total_records), table_cell_style), Paragraph("Total validated customer observations", table_cell_style)],
        [Paragraph("Feature Space", table_cell_style), Paragraph("Age, Income, Spending Score", table_cell_style), Paragraph("Key behavioral & financial dimensions", table_cell_style)],
        [Paragraph("Average Annual Income", table_cell_style), Paragraph(f"${avg_inc}k", table_cell_style), Paragraph("Mean income across observed sample", table_cell_style)],
        [Paragraph("Average Spending Score", table_cell_style), Paragraph(f"{avg_sp} / 100", table_cell_style), Paragraph("Customer spending rating index", table_cell_style)],
        [Paragraph("Average Customer Age", table_cell_style), Paragraph(f"{avg_age} years", table_cell_style), Paragraph("Mean chronological age of shoppers", table_cell_style)],
    ]
    t_data = Table(dataset_table_data, colWidths=[140, 110, 250])
    t_data.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')])
    ]))
    story.append(t_data)
    story.append(Spacer(1, 10))

    # Model Evaluation Metrics
    story.append(Paragraph("3. Unsupervised Model Architecture & Performance", heading2_style))
    n_k = model_bundle.get('n_clusters', 5)
    inertia = model_bundle.get('inertia', 'N/A')
    sil = model_bundle.get('silhouette_score', 'N/A')
    db = model_bundle.get('davies_bouldin_score', 'N/A')
    ch = model_bundle.get('calinski_harabasz_score', 'N/A')

    model_metrics_data = [
        [Paragraph("Hyperparameter / Metric", table_header_style), Paragraph("Empirical Value", table_header_style), Paragraph("Scientific Interpretation", table_header_style)],
        [Paragraph("Number of Clusters (K)", table_cell_style), Paragraph(str(n_k), table_cell_style), Paragraph("Identified via Elbow heuristic and Silhouette maximization", table_cell_style)],
        [Paragraph("Model Inertia (WCSS)", table_cell_style), Paragraph(str(inertia), table_cell_style), Paragraph("Sum of squared distances of samples to closest centroids", table_cell_style)],
        [Paragraph("Silhouette Coefficient", table_cell_style), Paragraph(str(sil), table_cell_style), Paragraph("Cluster separation & cohesion metric [-1.0 to +1.0]", table_cell_style)],
        [Paragraph("Davies-Bouldin Index", table_cell_style), Paragraph(str(db), table_cell_style), Paragraph("Average similarity measure of each cluster (lower is better)", table_cell_style)],
        [Paragraph("Calinski-Harabasz Index", table_cell_style), Paragraph(str(ch), table_cell_style), Paragraph("Variance ratio criterion (higher indicates dense clusters)", table_cell_style)],
    ]
    t_model = Table(model_metrics_data, colWidths=[140, 110, 250])
    t_model.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')])
    ]))
    story.append(t_model)
    story.append(Spacer(1, 10))

    # Cluster Profiles & Recommendations
    story.append(Paragraph("4. Segment Profiling & Business Action Plans", heading2_style))
    profiles = model_bundle.get('cluster_profiles', [])
    
    cluster_rows = [
        [
            Paragraph("ID", table_header_style),
            Paragraph("Segment Name", table_header_style),
            Paragraph("Size (%)", table_header_style),
            Paragraph("Centroid Profile", table_header_style),
            Paragraph("Strategic Action Plan", table_header_style)
        ]
    ]

    for p in profiles:
        cid = p.get('cluster_id')
        name = p.get('segment_name', f"Cluster {cid}")
        cnt = p.get('count', 0)
        pct = p.get('percentage', 0.0)
        centroid = p.get('centroid', {})
        
        centroid_str = ", ".join([f"{k.split(' ')[0]}: {v}" for k, v in centroid.items()])
        rec = p.get('recommendation', 'Standard marketing plan.')
        
        cluster_rows.append([
            Paragraph(f"C{cid}", table_cell_style),
            Paragraph(f"<b>{name}</b>", table_cell_style),
            Paragraph(f"{cnt} ({pct}%)", table_cell_style),
            Paragraph(centroid_str, table_cell_style),
            Paragraph(rec, table_cell_style)
        ])

    t_clusters = Table(cluster_rows, colWidths=[25, 120, 65, 120, 170])
    t_clusters.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')])
    ]))
    story.append(t_clusters)
    story.append(Spacer(1, 12))

    # Conclusion
    story.append(Paragraph("5. Academic Conclusion & Strategic Insights", heading2_style))
    conclusion_text = (
        "The empirical findings demonstrate clear behavioral separation among the mall customer cohort. "
        "Rather than treating customers uniformly, marketing teams can maximize return on marketing investment (ROMI) "
        "by allocating premium bespoke campaigns toward high-income high-spenders, while deploying payment flexibility and viral "
        "drops to younger high-spenders. The deployment of K-Means provides an objective, scalable foundation for continuous customer analytics."
    )
    story.append(Paragraph(conclusion_text, body_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    
    if output_path is None:
        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes
    else:
        buffer.close()
        return output_path
