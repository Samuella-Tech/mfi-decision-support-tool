import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import os
import warnings

# Suppress warnings for cleaner UI
warnings.filterwarnings("ignore")

# Page setup
st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")
st.title("📊 AI-Powered Credit Risk Dashboard for MFIs")
st.markdown("""
**Welcome!**
This dashboard helps microfinance teams assess borrower risk and repayment likelihood using uploaded data.

**To get started:**
Upload a CSV file with borrower details: `gender`, `age`, `income`, `loan_amount`, `occupation`, `loan_history`, `region`.

The dashboard will generate visual insights to support inclusive, data-driven lending decisions.
""")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload borrower data (.csv)", type=[
                                 "csv", "txt", "xlsx", "xls"])

# --- Load Data ---
required_columns = ["gender", "age", "income",
                    "loan_amount", "occupation", "loan_history", "region"]

try:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
    else:
        os.chdir(r"C:\\Users\\HP\\Desktop\\AI_CREDIT_RISK_PREDICTION")
        df = pd.read_csv("synthetic_borrowers_ghana.csv",
                         encoding="ISO-8859-1")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --- Column Validation ---
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    st.error(f"Missing required columns: {', '.join(missing_cols)}")
    st.stop()

# --- Feature Engineering ---
try:
    df["income_to_loan_ratio"] = df["income"] / df["loan_amount"]
    df["risk_score"] = np.clip(
        (df["income_to_loan_ratio"] * 0.3 + df["age"] * 0.01), 0, 1)
    df["risk_level"] = pd.cut(df["risk_score"], bins=[0, 0.3, 0.7, 1], labels=[
                              "Low", "Medium", "High"])
except KeyError as e:
    st.error(f"Feature engineering failed due to missing columns: {e}")
    st.stop()

# --- Sidebar Filters ---
st.sidebar.header("🔎 Filter Options")
selected_regions = st.sidebar.multiselect(
    "Select Region(s)", options=df["region"].unique())
if selected_regions:
    df = df[df["region"].isin(selected_regions)]

# --- Risk Distribution Chart ---


def create_risk_distribution_chart(df):
    color_map = {"Low": "green", "Medium": "gold", "High": "firebrick"}
    fig = px.histogram(
        df, x="risk_score", color="risk_level", nbins=30,
        title="Borrower Risk Score Distribution",
        labels={"risk_score": "Predicted Risk Score"},
        template="plotly_white",
        color_discrete_map=color_map,
        opacity=0.8
    )
    fig.update_layout(
        xaxis_title="Risk Score (0.0 - 1.0)",
        yaxis_title="Number of Borrowers",
        legend_title="Risk Level"
    )
    return fig


st.markdown("---")
st.subheader("📈 Risk Score Distribution Analysis")
st.plotly_chart(create_risk_distribution_chart(df), use_container_width=True)

# --- Regional Risk Chart ---


def create_regional_risk_chart(region_summary):
    fig = px.bar(
        region_summary,
        x="region",
        y="avg_risk_score",
        color="high_risk_pct",
        title="Average Risk Score by Region",
        labels={"avg_risk_score": "Avg Risk Score",
                "high_risk_pct": "High Risk %"},
        color_continuous_scale="Reds",
        template="plotly_white"
    )
    fig.update_layout(
        xaxis_title="Region",
        yaxis_title="Average Risk Score",
        coloraxis_colorbar=dict(title="High Risk %")
    )
    return fig


st.markdown("---")
st.subheader("🌍 Regional Risk Analysis")
region_summary = df.groupby("region").agg(
    avg_risk_score=("risk_score", "mean"),
    high_risk_pct=("risk_level", lambda x: (x == "High").mean() * 100)
).reset_index()
st.plotly_chart(create_regional_risk_chart(
    region_summary), use_container_width=True)
st.dataframe(region_summary.style.format(
    {"avg_risk_score": "{:.2f}", "high_risk_pct": "{:.1f}%"}))

# --- High-Risk Borrower Review ---
st.markdown("---")
colA, colB = st.columns([1, 2])

with colA:
    st.subheader("🚨 Top 10 High-Risk Borrowers")
    top_risk = df.sort_values(by="risk_score", ascending=False).head(10)
    selected_index = st.selectbox(
        "Select borrower index", options=top_risk.index.tolist())
    borrower = df.loc[selected_index]
    recommendation = "Review manually" if borrower["risk_level"] == "High" else "Likely Approve"
    st.write(
        f"Borrower #{selected_index} is **{borrower['risk_level']} risk**. Recommended action: **{recommendation}**.")

    st.subheader("🧠 Loan Decision Recommendation")
    top_indices = df.sort_values(
        by="risk_score", ascending=False).head(5).index.tolist()
    select_options = df.index.tolist()

    if top_indices and top_indices[0] in select_options:
        default_position = select_options.index(top_indices[0])
    elif select_options:
        st.warning(
            "No high-risk borrowers available in the current filter. Showing first borrower instead.")
        default_position = 0
    else:
        st.error("No borrowers available to review. Please adjust your filters.")
        default_position = None

    if select_options and default_position is not None:
        selected_index = st.selectbox(
            "Select borrower for review", options=select_options, index=default_position)
        borrower = df.loc[selected_index]
        recommendation = "Review manually" if borrower["risk_level"] == "High" else "Likely Approve"
        st.info(
            f"Borrower #{selected_index} is **{borrower['risk_level']}** risk (Score: {borrower['risk_score']:.3f}).\n\nRecommended action: **{recommendation}**."
        )

with colB:
    selected_level = st.selectbox("Filter by Risk Level", options=[
                                  "All", "Low", "Medium", "High"], index=3)
    filtered_df = df if selected_level == "All" else df[df["risk_level"]
                                                        == selected_level]
    st.write(
        f"Displaying {len(filtered_df)} borrowers in the **{selected_level}** group (Top 10):")
    st.dataframe(
        filtered_df[["age", "income", "loan_amount",
                     "risk_level", "risk_score", "income_to_loan_ratio"]]
        .sort_values(by="risk_score", ascending=(selected_level in ["Low", "All"]))
        .head(10),
        use_container_width=True
    )

# --- Summary Section ---
st.markdown("---")
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📋 Borrower Risk Overview")
    st.dataframe(df[["age", "income", "loan_amount",
                 "income_to_loan_ratio", "risk_score", "risk_level"]])

with col2:
    st.subheader("📊 Summary Stats")
    st.metric("Total Borrowers", len(df))
    st.metric("High Risk Borrowers", df[df["risk_level"] == "High"].shape[0])
    st.metric("Average Risk Score", round(df["risk_score"].mean(), 2))

# --- Footer ---
st.markdown("---")
st.markdown("""
📬 **Need help or want to contribute?**  
Visit the [GitHub repository](https://github.com/Samuella-Tech/) for documentation, updates, and collaboration.

_This dashboard supports ethical, inclusive financial decision-making in Ghana and beyond._
""")
