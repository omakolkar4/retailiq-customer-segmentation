import io
import datetime
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
    KeepTogether,
    HRFlowable,
)

def build_charts_for_pdf(
    customers_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    metrics_data: Dict[str, Any]
) -> Tuple[io.BytesIO, io.BytesIO]:
    """Render high-resolution Matplotlib charts for embedded ReportLab flowables."""
    # Chart 1: Customer Distribution and Revenue Contribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    
    # Left: Customer Distribution Bar
    persona_counts = profiles_df.set_index("Persona")["Total_Customers"]
    persona_counts.plot(kind="bar", ax=ax1, color="#1e3a8a", edgecolor="#0f172a")
    ax1.set_title("Customer Count by Persona", fontsize=11, fontweight="bold", pad=8)
    ax1.set_ylabel("Customers", fontsize=9)
    ax1.set_xlabel("")
    ax1.tick_params(axis='x', rotation=25, labelsize=8)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # Right: Revenue Contribution Bar
    persona_rev = profiles_df.set_index("Persona")["Total_Revenue"]
    persona_rev.plot(kind="bar", ax=ax2, color="#0284c7", edgecolor="#0f172a")
    ax2.set_title("Revenue Contribution by Persona ($)", fontsize=11, fontweight="bold", pad=8)
    ax2.set_ylabel("Total Spend ($)", fontsize=9)
    ax2.set_xlabel("")
    ax2.tick_params(axis='x', rotation=25, labelsize=8)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    fig.tight_layout()
    chart1_buf = io.BytesIO()
    fig.savefig(chart1_buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    chart1_buf.seek(0)

    # Chart 2: Elbow & Silhouette Plot
    fig2, (bx1, bx2) = plt.subplots(1, 2, figsize=(10, 3.8))
    elbow_points = metrics_data.get("elbow_curve", [])
    if elbow_points:
        ks = [pt["k"] for pt in elbow_points]
        inertias = [pt["inertia"] for pt in elbow_points]
        sils = [pt["silhouette"] for pt in elbow_points]

        bx1.plot(ks, inertias, marker="o", color="#0f766e", linewidth=2)
        bx1.set_title("K-Means Elbow Method (Inertia)", fontsize=11, fontweight="bold", pad=8)
        bx1.set_xlabel("Number of Clusters (K)", fontsize=9)
        bx1.set_ylabel("Inertia", fontsize=9)
        bx1.grid(True, linestyle="--", alpha=0.5)

        bx2.plot(ks, sils, marker="s", color="#7c3aed", linewidth=2)
        bx2.set_title("Silhouette Score vs K", fontsize=11, fontweight="bold", pad=8)
        bx2.set_xlabel("Number of Clusters (K)", fontsize=9)
        bx2.set_ylabel("Silhouette Score", fontsize=9)
        bx2.grid(True, linestyle="--", alpha=0.5)
    else:
        bx1.text(0.5, 0.5, "Elbow data unavailable", ha="center")
        bx2.text(0.5, 0.5, "Silhouette data unavailable", ha="center")

    fig2.tight_layout()
    chart2_buf = io.BytesIO()
    fig2.savefig(chart2_buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig2)
    chart2_buf.seek(0)

    return chart1_buf, chart2_buf

def generate_pdf_report(
    kpis: Dict[str, Any],
    profiles_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    models_data: Dict[str, Any],
    validation_data: Dict[str, Any],
    filename: str = "Online Retail Dataset"
) -> bytes:
    """Generate executive-grade multi-page PDF Report using ReportLab."""
    pdf_buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=landscape(A4),
        rightMargin=30,
        leftMargin=30,
        topMargin=25,
        bottomMargin=25,
    )

    styles = getSampleStyleSheet()
    
    # Custom Brand Typography
    primary_color = colors.HexColor("#0f172a") # Slate 900
    accent_blue = colors.HexColor("#1e3a8a")   # Blue 900
    brand_teal = colors.HexColor("#0f766e")    # Teal 700

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=20,
        leading=24,
        textColor=accent_blue,
        alignment=0, # Left-aligned
        fontName="Helvetica-Bold",
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
    )
    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Heading1"],
        fontSize=13,
        leading=16,
        textColor=primary_color,
        fontName="Helvetica-Bold",
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1e293b"),
    )

    story = []

    # Title & Metadata Banner
    today_str = datetime.datetime.now().strftime("%B %d, %Y")
    story.append(Paragraph("RetailIQ — Customer Intelligence & Segmentation Platform", title_style))
    story.append(Paragraph(f"Executive Retail Analytics Report  |  Source: {filename}  |  Generated: {today_str}", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=accent_blue, spaceBefore=2, spaceAfter=10))

    # Section 1: Executive KPI Summary Table
    story.append(Paragraph("1. Executive Summary & Core Retail KPIs", h1_style))
    kpi_table_data = [
        ["Total Customers", "Total Revenue", "Avg Customer Spend", "Median Spend", "Avg Invoices/Customer", "Retention Rate"],
        [
            f"{kpis.get('total_customers', 0):,}",
            f"${kpis.get('total_revenue', 0.0):,.2f}",
            f"${kpis.get('avg_customer_spend', 0.0):,.2f}",
            f"${kpis.get('median_spend', 0.0):,.2f}",
            f"{kpis.get('avg_frequency', 0.0):.2f}",
            f"{kpis.get('retention_rate', 0.0):.1f}%",
        ]
    ]
    kpi_table = Table(kpi_table_data, colWidths=[130, 130, 130, 130, 130, 130])
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#f8fafc")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 12))

    # Section 2: Segment Profiles Table
    story.append(Paragraph("2. Customer Segment Behavioral Profiles", h1_style))
    seg_headers = ["Persona", "Customers", "% Share", "Revenue ($)", "% Rev", "Avg Recency", "Avg Freq", "Avg Spend ($)", "Risk Level"]
    seg_rows = [seg_headers]
    for _, r in profiles_df.iterrows():
        seg_rows.append([
            str(r["Persona"]),
            f"{int(r['Total_Customers']):,}",
            f"{r.get('Pct_Customers', 0.0):.1f}%",
            f"${r['Total_Revenue']:,.2f}",
            f"{r.get('Pct_Revenue', 0.0):.1f}%",
            f"{r['Average_Recency']:.1f} days",
            f"{r['Average_Frequency']:.2f}",
            f"${r['Average_Monetary']:,.2f}",
            str(r.get("Risk_Level", "Normal")),
        ])
    
    seg_table = Table(seg_rows, colWidths=[150, 65, 55, 95, 55, 75, 65, 85, 85])
    seg_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 1), (0, -1), "LEFT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(seg_table)
    story.append(Spacer(1, 10))

    # Embed Chart 1
    chart1_buf, chart2_buf = build_charts_for_pdf(customers_df, profiles_df, models_data)
    story.append(Image(chart1_buf, width=720, height=200))

    # Page 2: Machine Learning Evaluation & Clustering Stability
    story.append(PageBreak())
    story.append(Paragraph("3. Multi-Model Machine Learning Benchmarking", h1_style))
    story.append(Paragraph(
        "K-Means, Agglomerative Hierarchical, and Gaussian Mixture Models were trained on the identical log-transformed, "
        "StandardScaler-normalized RFM vectors. Metrics include Silhouette Score (cohesion vs separation), Davies-Bouldin Index "
        "(cluster compactness; lower is superior), and Calinski-Harabasz Score (variance ratio; higher is superior).",
        body_style
    ))
    story.append(Spacer(1, 6))

    model_headers = ["Algorithm", "Clusters (K)", "Silhouette", "Davies-Bouldin", "Calinski-Harabasz", "Inertia", "Latency"]
    model_rows = [model_headers]
    for m in models_data.get("models", []):
        inertia_str = f"{m['inertia']:,.1f}" if m.get("inertia") is not None else "N/A (Non-Centroid)"
        model_rows.append([
            m["model_name"],
            str(m["n_clusters"]),
            f"{m['silhouette_score']:.4f}",
            f"{m['davies_bouldin_index']:.4f}",
            f"{m['calinski_harabasz_score']:,.1f}",
            inertia_str,
            f"{m['execution_time_ms']:.1f} ms",
        ])
    
    model_table = Table(model_rows, colWidths=[170, 70, 85, 95, 110, 110, 80])
    model_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 1), (0, -1), "LEFT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(model_table)
    story.append(Spacer(1, 8))

    rec_box = [
        [f"Recommended Model: {models_data.get('recommended_model', 'K-Means')}", f"Rationale: {models_data.get('recommendation_reason', '')}"]
    ]
    rec_table = Table(rec_box, colWidths=[200, 520])
    rec_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eff6ff")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#1e3a8a")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#93c5fd")),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(rec_table)
    story.append(Spacer(1, 10))

    # Embed Chart 2: Elbow & Silhouette curves
    story.append(Image(chart2_buf, width=720, height=200))
    story.append(Spacer(1, 10))

    # Section 4: Clustering Stability Validation
    story.append(Paragraph("4. Unsupervised Clustering Stability (Repeated Subsampling)", h1_style))
    val_notes = (
        f"Conducted {validation_data.get('n_iterations', 10)} bootstrap iterations at "
        f"{int(validation_data.get('subsample_ratio', 0.8) * 100)}% subsampling without replacement. "
        f"Mean Adjusted Rand Index (ARI): <b>{validation_data.get('mean_ari', 0.0):.4f}</b> (±{validation_data.get('std_ari', 0.0):.4f}). "
        f"Mean Silhouette: <b>{validation_data.get('mean_silhouette', 0.0):.4f}</b>. "
        "High ARI demonstrates that cluster boundaries are robust against sampling variance."
    )
    story.append(Paragraph(val_notes, body_style))

    # Page 3: Marketing Intelligence & Action Playbooks
    story.append(PageBreak())
    story.append(Paragraph("5. Strategic Marketing Playbooks & Campaign Blueprint", h1_style))

    mkt_rows = [["Customer Persona", "Core Behavioral Drivers", "Recommended Marketing Campaign", "Target Channels"]]
    for _, r in profiles_df.iterrows():
        action_text = str(r["Marketing_Action"])
        mkt_rows.append([
            str(r["Persona"]),
            f"Recency: {r['Average_Recency']:.0f}d | Spend: ${r['Average_Monetary']:,.0f} | Freq: {r['Average_Frequency']:.1f}",
            action_text,
            "Email, Push, SMS, Dedicated Concierge" if "VIP" in str(r["Persona"]) else "Email, Loyalty Drips, Retargeting",
        ])
    
    mkt_table = Table(mkt_rows, colWidths=[150, 160, 270, 140])
    mkt_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(mkt_table)
    story.append(Spacer(1, 15))

    # Governance Note
    governance_text = (
        "<b>Academic & Governance Note:</b> Persona titles and marketing recommendations are rule-based domain interpretations "
        "of unsupervised cluster centroids and do not constitute supervised outcome predictions. Campaign efficacy must be "
        "empirically validated using A/B holdout testing prior to enterprise resource allocation."
    )
    story.append(Paragraph(governance_text, body_style))

    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()
