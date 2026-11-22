import io
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(
    page_title="Retail Customer Insights",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PERSONA_ACTIONS = {
    "VIP High-Value Customers": "Offer exclusive rewards, VIP benefits, and early access to new products.",
    "Loyal Customers": "Offer loyalty rewards and personalized product recommendations.",
    "Regular Customers": "Send personalized offers to encourage repeat purchases.",
    "At-Risk Customers": "Send win-back discounts and re-engagement campaigns.",
}


def read_transactions(file):
    """Read CSV or Excel transaction data from an uploaded file."""
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    raise ValueError("Please upload a CSV or Excel file.")


def normalize_columns(df):
    """Map common Online Retail column names to a consistent schema."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    aliases = {
        "CustomerID": ["CustomerID", "Customer ID", "customer_id", "customerid"],
        "InvoiceNo": ["InvoiceNo", "Invoice No", "Invoice", "invoice_no", "invoiceno"],
        "InvoiceDate": ["InvoiceDate", "Invoice Date", "date", "invoice_date"],
        "Quantity": ["Quantity", "quantity", "Qty"],
        "UnitPrice": ["UnitPrice", "Unit Price", "Price", "unit_price", "price"],
    }
    rename = {}
    for canonical, candidates in aliases.items():
        for candidate in candidates:
            if candidate in df.columns:
                rename[candidate] = canonical
                break
    df = df.rename(columns=rename)

    required = ["CustomerID", "InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
            + ". Expected transaction data with CustomerID, InvoiceNo, "
              "InvoiceDate, Quantity, and UnitPrice."
        )

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    for col in ["Quantity", "UnitPrice", "CustomerID"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    return df


def build_rfm(raw):
    """Clean transactions and calculate Recency, Frequency, and Monetary per customer."""
    df = normalize_columns(raw)
    initial_rows = len(df)

    df = df.dropna(subset=["CustomerID", "InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"])
    df = df[~df["InvoiceNo"].str.upper().str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

    if df.empty:
        raise ValueError("No valid transaction rows remain after cleaning.")

    reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        LastPurchase=("InvoiceDate", "max"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalAmount", "sum"),
    )
    rfm["Recency"] = (reference_date - rfm["LastPurchase"]).dt.days
    rfm = rfm.reset_index()
    rfm = rfm[["CustomerID", "Recency", "Frequency", "Monetary"]]
    rfm = rfm.replace([np.inf, -np.inf], np.nan).dropna()
    rfm = rfm[(rfm["Recency"] >= 0) & (rfm["Frequency"] > 0) & (rfm["Monetary"] > 0)]
    if len(rfm) < 4:
        raise ValueError("At least 4 customers with valid RFM values are required for 4-cluster segmentation.")

    return df, rfm, initial_rows, reference_date


def train_segmentation(rfm, k=4):
    """Train K-Means on log-transformed RFM and return metrics, PCA, and segment profiles."""
    features = ["Recency", "Frequency", "Monetary"]
    model_data = rfm[features].copy()
    # RFM values are often highly skewed; log1p reduces the impact of extreme values.
    model_log = np.log1p(model_data)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(model_log)

    max_k = min(10, len(rfm) - 1)
    ks = list(range(2, max_k + 1))
    inertia, silhouettes = [], []
    for n_clusters in ks:
        candidate = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = candidate.fit_predict(scaled)
        inertia.append(float(candidate.inertia_))
        if len(np.unique(labels)) > 1:
            silhouettes.append(float(silhouette_score(scaled, labels, sample_size=min(2000, len(scaled)), random_state=42)))
        else:
            silhouettes.append(np.nan)

    k = min(k, len(rfm) - 1)
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = model.fit_predict(scaled)

    result = rfm.copy()
    result["Cluster"] = clusters
    profiles = result.groupby("Cluster").agg(
        Total_Customers=("CustomerID", "count"),
        Average_Recency=("Recency", "mean"),
        Average_Frequency=("Frequency", "mean"),
        Average_Monetary=("Monetary", "mean"),
        Median_Monetary=("Monetary", "median"),
    ).round(2)

    # Persona names are assigned from cluster behaviour, not hard-coded cluster numbers.
    # Rank clusters by spending/frequency and recency (lower recency means more recent).
    ranking = profiles.copy()
    ranking["spend_rank"] = ranking["Average_Monetary"].rank(pct=True)
    ranking["freq_rank"] = ranking["Average_Frequency"].rank(pct=True)
    ranking["recent_rank"] = (1 - ranking["Average_Recency"].rank(pct=True))
    ranking["value_score"] = (
        ranking["spend_rank"] * 0.45
        + ranking["freq_rank"] * 0.35
        + ranking["recent_rank"] * 0.20
    )

    # Highest value cluster is VIP; remaining high-frequency/recent cluster is Loyal;
    # highest-recency/low-frequency cluster is At-Risk; remaining cluster(s) are Regular.
    value_order = ranking.sort_values("value_score", ascending=False).index.tolist()
    risk_order = profiles.sort_values(
        ["Average_Recency", "Average_Frequency", "Average_Monetary"],
        ascending=[False, True, True]
    ).index.tolist()

    persona_map = {}
    if len(value_order) >= 1:
        persona_map[value_order[0]] = "VIP High-Value Customers"
    if len(value_order) >= 2:
        persona_map[value_order[1]] = "Loyal Customers"
    if len(risk_order) >= 1 and risk_order[0] not in persona_map:
        persona_map[risk_order[0]] = "At-Risk Customers"
    for cluster_id in profiles.index:
        if cluster_id not in persona_map:
            persona_map[cluster_id] = "Regular Customers"

    # With more than 4 clusters, add unique neutral labels to avoid duplicate persona categories.
    for cluster_id in profiles.index:
        if list(persona_map.values()).count(persona_map[cluster_id]) > 1:
            persona_map[cluster_id] = f"Customer Segment {cluster_id}"

    result["Persona"] = result["Cluster"].map(persona_map)
    result["Marketing_Action"] = result["Persona"].map(PERSONA_ACTIONS).fillna(
        "Review this segment's purchase behaviour and design a targeted campaign."
    )
    profiles["Persona"] = profiles.index.map(persona_map)
    profiles["Marketing_Action"] = profiles["Persona"].map(PERSONA_ACTIONS).fillna(
        "Review this segment's purchase behaviour and design a targeted campaign."
    )

    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(scaled)
    result["PCA1"] = components[:, 0]
    result["PCA2"] = components[:, 1]

    metrics = pd.DataFrame({
        "K": ks,
        "Inertia": inertia,
        "Silhouette Score": silhouettes,
    })
    return result, profiles, metrics, model, scaler, persona_map, pca, scaled, features


def money(value):
    return f"{value:,.2f}"

def create_pdf_report(customers_df, profiles_df, metrics_df):
    import io
    import matplotlib.pyplot as plt

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        Image,
        PageBreak,
    )

    pdf_buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=landscape(A4),
        rightMargin=35,
        leftMargin=35,
        topMargin=35,
        bottomMargin=35,
    )

    styles = getSampleStyleSheet()
    story = []

    # Report title
    story.append(Paragraph(
        "Retail Customer Segmentation Report",
        styles["Title"]
    ))
    story.append(Paragraph(
        "Customer Segmentation for a Retail Chain",
        styles["Heading2"]
    ))
    story.append(Spacer(1, 18))

    # Business summary
    summary = [
        ["Metric", "Value"],
        ["Total Customers", f"{customers_df['CustomerID'].nunique():,}"],
        ["Total Spending", f"{customers_df['Monetary'].sum():,.2f}"],
        ["Average Spending", f"{customers_df['Monetary'].mean():,.2f}"],
        ["Average Frequency", f"{customers_df['Frequency'].mean():.2f}"],
    ]

    table = Table(summary, colWidths=[230, 180])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24476B")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 7),
    ]))

    story.append(Paragraph("1. Business Overview", styles["Heading1"]))
    story.append(table)
    story.append(Spacer(1, 18))

    # Customer distribution chart
    story.append(Paragraph(
        "2. Customer Distribution by Persona",
        styles["Heading1"]
    ))

    counts = customers_df.groupby("Persona")["CustomerID"].nunique()

    fig, ax = plt.subplots(figsize=(9, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_ylabel("Number of Customers")
    ax.set_xlabel("Customer Persona")
    ax.set_title("Customer Distribution by Persona")
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()

    chart_buffer = io.BytesIO()
    fig.savefig(chart_buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    chart_buffer.seek(0)

    story.append(Image(chart_buffer, width=500, height=230))
    story.append(Spacer(1, 18))

    # Segment profile table
    story.append(PageBreak())
    story.append(Paragraph(
        "3. Customer Segment Profiles",
        styles["Heading1"]
    ))

    profile_data = [[
        "Persona", "Customers", "Avg Recency",
        "Avg Frequency", "Avg Spending"
    ]]

    for _, row in profiles_df.iterrows():
        profile_data.append([
            str(row["Persona"]),
            f"{int(row['Total_Customers']):,}",
            f"{row['Average_Recency']:.2f}",
            f"{row['Average_Frequency']:.2f}",
            f"{row['Average_Monetary']:,.2f}",
        ])

    profile_table = Table(
        profile_data,
        colWidths=[175, 75, 85, 90, 100],
        repeatRows=1,
    )
    profile_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24476B")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))

    story.append(profile_table)
    story.append(Spacer(1, 18))

    # Marketing recommendations
    story.append(Paragraph(
        "4. Recommended Marketing Actions",
        styles["Heading1"]
    ))

    action_data = [["Persona", "Marketing Action"]]

    for _, row in profiles_df.iterrows():
        action_data.append([
            str(row["Persona"]),
            str(row["Marketing_Action"]),
        ])

    action_table = Table(action_data, colWidths=[175, 410], repeatRows=1)
    action_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24476B")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))

    story.append(action_table)
    story.append(Spacer(1, 18))

    # Model evaluation
    story.append(Paragraph(
        "5. Model Evaluation",
        styles["Heading1"]
    ))

    eval_data = [["K", "Inertia", "Silhouette Score"]]

    for _, row in metrics_df.iterrows():
        eval_data.append([
            str(int(row["K"])),
            f"{row['Inertia']:.2f}",
            f"{row['Silhouette Score']:.4f}",
        ])

    eval_table = Table(eval_data, repeatRows=1)
    eval_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24476B")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))

    story.append(eval_table)
    story.append(Spacer(1, 15))
    story.append(Paragraph(
        "Marketing actions are recommendations based on customer segments. "
        "They do not prove campaign effectiveness.",
        styles["BodyText"]
    ))

    doc.build(story)
    pdf_buffer.seek(0)

    return pdf_buffer.getvalue()



st.title("🛍️ Retail Customer Insights Dashboard")
st.write(
    "A practical retail analytics application using RFM analysis, K-Means clustering, "
    "customer personas, PCA visualization, and marketing recommendations."
)

with st.sidebar:
    st.header("Data & Settings")
    uploaded_file = st.file_uploader(
        "Upload transaction data",
        type=["xlsx", "xls", "csv"],
        help="Expected columns: CustomerID, InvoiceNo, InvoiceDate, Quantity, UnitPrice.",
    )
    st.caption("For the Online Retail dataset, use the original transaction-level Excel/CSV file.")
    cluster_count = st.slider("Number of clusters (K)", min_value=2, max_value=6, value=4, step=1)
    st.caption("The model is trained on the currently loaded transaction file.")

if uploaded_file is None:
    st.info("Upload your Online Retail.xlsx or a compatible transaction CSV/Excel file in the sidebar to start.")
    st.markdown(
        """
        **Expected transaction columns**
        - `CustomerID`
        - `InvoiceNo`
        - `InvoiceDate`
        - `Quantity`
        - `UnitPrice`

        Optional columns such as `Description`, `StockCode`, and `Country` are allowed.
        """
    )
    st.stop()

try:
    raw_transactions = read_transactions(uploaded_file)
    clean_transactions, rfm, initial_rows, reference_date = build_rfm(raw_transactions)
    (
        customers, profiles, metrics, kmeans, scaler, persona_map, pca, scaled_data, features
    ) = train_segmentation(rfm, cluster_count)
except Exception as exc:
    st.error(f"Could not process this file: {exc}")
    st.stop()

st.success(f"Processed {len(clean_transactions):,} valid transaction rows into {len(customers):,} customer records.")
st.caption(
    f"Reference date used for Recency: {reference_date:%Y-%m-%d}. "
    f"Rows before cleaning: {initial_rows:,}; rows after cleaning: {len(clean_transactions):,}."
)

with st.sidebar:
    all_personas = sorted(customers["Persona"].dropna().unique().tolist())
    selected_personas = st.multiselect("Filter customer personas", all_personas, default=all_personas)

filtered = customers[customers["Persona"].isin(selected_personas)].copy()
filtered_profiles = profiles[profiles["Persona"].isin(selected_personas)].copy()

# KPI row
k1, k2, k3, k4 = st.columns(4)
k1.metric("Customers", f"{filtered['CustomerID'].nunique():,}")
k2.metric("Total customer spend", money(filtered["Monetary"].sum()))
k3.metric("Average spend / customer", money(filtered["Monetary"].mean()))
k4.metric("Average invoices / customer", f"{filtered['Frequency'].mean():.2f}")

tabs = st.tabs([
    "Overview",
    "Customer Explorer",
    "RFM & Segments",
    "Model Evaluation",
    "Export Results",
])

with tabs[0]:
    st.subheader("Customer distribution by persona")
    distribution = filtered.groupby("Persona", as_index=False).agg(Customers=("CustomerID", "nunique"))
    if not distribution.empty:
        fig = px.bar(
            distribution, x="Persona", y="Customers", color="Persona",
            text="Customers", title="Number of Customers in Each Persona"
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Customers")
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Average monetary value")
        spend = filtered.groupby("Persona", as_index=False).agg(
            Average_Monetary=("Monetary", "mean")
        )
        fig = px.bar(spend, x="Persona", y="Average_Monetary", color="Persona",
                     title="Average Customer Spending by Persona")
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Average spend")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Average purchase frequency")
        freq = filtered.groupby("Persona", as_index=False).agg(
            Average_Frequency=("Frequency", "mean")
        )
        fig = px.bar(freq, x="Persona", y="Average_Frequency", color="Persona",
                     title="Average Invoice Count by Persona")
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Average invoices")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Segment profile and recommended action")
    st.dataframe(filtered_profiles.reset_index(drop=True), use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("Search for a customer")
    customer_ids = sorted(customers["CustomerID"].unique().tolist())
    chosen_id = st.selectbox("Select Customer ID", customer_ids, format_func=lambda x: str(int(x)) if float(x).is_integer() else str(x))
    selected = customers[customers["CustomerID"] == chosen_id].iloc[0]
    a, b, c, d = st.columns(4)
    a.metric("Recency (days)", f"{int(selected['Recency'])}")
    b.metric("Invoices", f"{int(selected['Frequency'])}")
    c.metric("Monetary value", money(selected["Monetary"]))
    d.metric("Cluster", str(int(selected["Cluster"])))
    st.markdown(f"### Persona: {selected['Persona']}")
    st.info(f"**Recommended marketing action:** {selected['Marketing_Action']}")
    st.dataframe(pd.DataFrame([selected]), use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader("RFM scatter plot")
    x_axis = st.selectbox("X-axis", ["Recency", "Frequency", "Monetary"], index=1)
    y_axis = st.selectbox("Y-axis", ["Monetary", "Frequency", "Recency"], index=0)
    fig = px.scatter(
        filtered, x=x_axis, y=y_axis, color="Persona",
        hover_data=["CustomerID", "Recency", "Frequency", "Monetary"],
        title=f"{y_axis} vs {x_axis}"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("PCA visualization")
    fig = px.scatter(
        filtered, x="PCA1", y="PCA2", color="Persona",
        hover_data=["CustomerID", "Recency", "Frequency", "Monetary"],
        title="Customer Segments in 2D PCA Space"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cluster profiles")
    st.dataframe(filtered_profiles.reset_index(drop=True), use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("Choosing the number of clusters")
    left, right = st.columns(2)
    with left:
        fig = px.line(metrics, x="K", y="Inertia", markers=True, title="Elbow Method")
        fig.update_layout(xaxis_title="Number of clusters (K)", yaxis_title="Inertia")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.line(metrics, x="K", y="Silhouette Score", markers=True, title="Silhouette Score")
        fig.update_layout(xaxis_title="Number of clusters (K)", yaxis_title="Silhouette score")
        st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "The current dashboard trains the selected K-Means model on the uploaded dataset. "
        "Use both the elbow curve and silhouette score, plus business interpretability, when choosing K."
    )
    st.dataframe(metrics, use_container_width=True, hide_index=True)

    st.subheader("Hierarchical clustering comparison")
    if len(scaled_data) >= 4:
        sample_n = min(1500, len(scaled_data))
        sample_idx = np.linspace(0, len(scaled_data) - 1, sample_n, dtype=int)
        sample_scaled = scaled_data[sample_idx]
        if sample_n > 1:
            hierarchical = AgglomerativeClustering(n_clusters=cluster_count, linkage="ward")
            hier_labels = hierarchical.fit_predict(sample_scaled)
            comparison = pd.DataFrame({
                "KMeans_Cluster": customers.iloc[sample_idx]["Cluster"].to_numpy(),
                "Hierarchical_Cluster": hier_labels,
            })
            st.write("Comparison on a sample of customers (cluster numbers are arbitrary across methods):")
            st.dataframe(pd.crosstab(comparison["KMeans_Cluster"], comparison["Hierarchical_Cluster"]),
                         use_container_width=True)
    else:
        st.info("Not enough customers for hierarchical clustering.")

with tabs[4]:
    st.subheader("Download customer results")

    st.write(
        "The table includes customer-level RFM values, "
        "cluster labels, personas, PCA coordinates, and marketing actions."
    )

    st.dataframe(filtered, use_container_width=True, hide_index=True)

    # Download customer results as CSV
    csv_bytes = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download filtered customer results (CSV)",
        data=csv_bytes,
        file_name="customer_segmentation_results.csv",
        mime="text/csv",
    )

    # Download segment summary as CSV
    profile_csv = filtered_profiles.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download segment summary (CSV)",
        data=profile_csv,
        file_name="customer_segment_summary.csv",
        mime="text/csv",
    )

    # Download report as PDF
    st.divider()
    st.subheader("Download PDF Report")

    pdf_report = create_pdf_report(
        filtered,
        filtered_profiles,
        metrics
    )

    st.download_button(
        label="📄 Download Customer Segmentation PDF",
        data=pdf_report,
        file_name="Retail_Customer_Segmentation_Report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


st.divider()
st.caption(
    "Academic prototype: recommendations are rule-based suggestions, not proof of campaign impact. "
    "The clustering model is refitted whenever a new transaction file is uploaded."
)
