import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import poisson


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Medication Demand & Stock Analysis",
    page_icon="💊",
    layout="wide"
)


# ==========================================================
# TITLE
# ==========================================================

st.title("💊 Medication Demand and Stock Analysis")

st.write(
    """
    This dashboard analyses historical medication demand,
    models demand using a Poisson distribution and stochastic
    process, and identifies potential stock shortages.
    """
)


# ==========================================================
# DEMO DATA GENERATOR
# ==========================================================

def generate_demo_data():

    np.random.seed(42)

    dates = pd.date_range(
        start="2025-01-01",
        periods=120,
        freq="D"
    )

    medicines = [
        "Paracetamol",
        "Amoxicillin",
        "Ibuprofen",
        "Cetirizine"
    ]

    rows = []

    base_demand = {
        "Paracetamol": 45,
        "Amoxicillin": 25,
        "Ibuprofen": 35,
        "Cetirizine": 20
    }

    for medicine in medicines:

        for date in dates:

            demand = np.random.poisson(
                base_demand[medicine]
            )

            stock_received = np.random.choice(
                [0, 30, 50, 80],
                p=[0.55, 0.20, 0.15, 0.10]
            )

            rows.append({
                "date": date,
                "medicine": medicine,
                "demand": demand,
                "stock_received": stock_received
            })

    return pd.DataFrame(rows)


# ==========================================================
# DATA INPUT
# ==========================================================

st.sidebar.header("📂 Data Input")

uploaded_file = st.sidebar.file_uploader(
    "Upload medication CSV",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

        st.sidebar.success(
            "Dataset uploaded successfully!"
        )

    except Exception:

        st.sidebar.error(
            "Unable to read the CSV file."
        )

        st.stop()

else:

    st.sidebar.info(
        "No dataset uploaded. Demo data is being used."
    )

    df = generate_demo_data()


# ==========================================================
# DATA VALIDATION
# ==========================================================

required_columns = [
    "date",
    "medicine",
    "demand",
    "stock_received"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    st.error(
        "Missing columns: "
        + ", ".join(missing_columns)
    )

    st.info(
        """
        Your CSV must contain:

        date, medicine, demand, stock_received
        """
    )

    st.stop()


# ==========================================================
# DATA PREPROCESSING
# ==========================================================

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df["demand"] = pd.to_numeric(
    df["demand"],
    errors="coerce"
)

df["stock_received"] = pd.to_numeric(
    df["stock_received"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "date",
        "medicine",
        "demand",
        "stock_received"
    ]
)

df = df.sort_values("date")


# ==========================================================
# MEDICINE FILTER
# ==========================================================

st.sidebar.subheader("💊 Medicine Filter")

medicine_list = sorted(
    df["medicine"].unique()
)

selected_medicine = st.sidebar.selectbox(
    "Select medicine",
    ["All Medicines"] + medicine_list
)


if selected_medicine == "All Medicines":

    analysis_df = df.copy()

else:

    analysis_df = df[
        df["medicine"] == selected_medicine
    ].copy()


# ==========================================================
# KEY STATISTICS
# ==========================================================

st.subheader("📊 Key Statistics")

total_demand = int(
    analysis_df["demand"].sum()
)

average_demand = (
    analysis_df["demand"].mean()
)

maximum_demand = int(
    analysis_df["demand"].max()
)

total_received = int(
    analysis_df["stock_received"].sum()
)


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Demand",
    total_demand
)

col2.metric(
    "Average Demand",
    f"{average_demand:.2f}"
)

col3.metric(
    "Maximum Demand",
    maximum_demand
)

col4.metric(
    "Stock Received",
    total_received
)


# ==========================================================
# DATA PREVIEW
# ==========================================================

st.subheader("📋 Dataset Preview")

st.dataframe(
    analysis_df,
    use_container_width=True
)


# ==========================================================
# DEMAND TREND
# ==========================================================

st.subheader("📈 Medication Demand Trend")

daily_demand = (
    analysis_df
    .groupby("date")["demand"]
    .sum()
    .reset_index()
)

fig, ax = plt.subplots()

ax.plot(
    daily_demand["date"],
    daily_demand["demand"],
    marker="o",
    markersize=3
)

ax.set_xlabel("Date")
ax.set_ylabel("Demand")

ax.set_title(
    "Medication Demand Over Time"
)

ax.grid(True, alpha=0.3)

plt.xticks(rotation=45)

st.pyplot(fig)

plt.close(fig)


# ==========================================================
# MEDICINE-WISE DEMAND
# ==========================================================

st.subheader("💊 Medicine-wise Demand")

medicine_demand = (
    df
    .groupby("medicine")["demand"]
    .sum()
    .sort_values(ascending=False)
)

fig, ax = plt.subplots()

medicine_demand.plot(
    kind="bar",
    ax=ax
)

ax.set_xlabel("Medicine")
ax.set_ylabel("Total Demand")

ax.set_title(
    "Total Demand by Medicine"
)

plt.xticks(rotation=45)

st.pyplot(fig)

plt.close(fig)


# ==========================================================
# DESCRIPTIVE STATISTICS
# ==========================================================

st.subheader("📐 Demand Statistics")

mean_demand = analysis_df["demand"].mean()

variance_demand = analysis_df["demand"].var()

std_demand = analysis_df["demand"].std()

col1, col2, col3 = st.columns(3)

col1.metric(
    "Mean Demand",
    f"{mean_demand:.2f}"
)

col2.metric(
    "Variance",
    f"{variance_demand:.2f}"
)

col3.metric(
    "Standard Deviation",
    f"{std_demand:.2f}"
)


# ==========================================================
# POISSON DISTRIBUTION
# ==========================================================

st.subheader("🎲 Poisson Distribution Analysis")

lambda_value = mean_demand

st.write(
    f"""
    Estimated Poisson parameter:

    **λ = {lambda_value:.3f}**
    """
)

max_demand = int(
    analysis_df["demand"].max()
)

x = np.arange(
    0,
    max_demand + 1
)

poisson_probability = poisson.pmf(
    x,
    lambda_value
)


# Observed probability

observed_frequency = (
    analysis_df["demand"]
    .value_counts()
    .reindex(x, fill_value=0)
)

observed_probability = (
    observed_frequency /
    len(analysis_df)
)


fig, ax = plt.subplots()

ax.bar(
    x,
    observed_probability,
    alpha=0.6,
    label="Observed Demand"
)

ax.plot(
    x,
    poisson_probability,
    marker="o",
    label="Poisson Model"
)

ax.set_xlabel("Demand")
ax.set_ylabel("Probability")

ax.set_title(
    "Observed Demand vs Poisson Distribution"
)

ax.legend()

ax.grid(True, alpha=0.3)

st.pyplot(fig)

plt.close(fig)


# ==========================================================
# STOCHASTIC PROBABILITY
# ==========================================================

st.subheader("🔮 Demand Probability")

requested_demand = st.number_input(
    "Enter a demand value",
    min_value=0,
    value=int(round(lambda_value)),
    step=1
)

probability = poisson.pmf(
    requested_demand,
    lambda_value
)

st.metric(
    f"Probability of exactly {requested_demand} units",
    f"{probability * 100:.2f}%"
)


# ==========================================================
# HIGH DEMAND PROBABILITY
# ==========================================================

st.subheader("⚠️ High-Demand Probability")

threshold = st.number_input(
    "High-demand threshold",
    min_value=0,
    value=int(round(lambda_value * 1.25)),
    step=1
)

high_demand_probability = (
    1 -
    poisson.cdf(
        threshold - 1,
        lambda_value
    )
)

st.metric(
    f"Probability of demand ≥ {threshold}",
    f"{high_demand_probability * 100:.2f}%"
)


# ==========================================================
# STOCK ANALYSIS
# ==========================================================

st.subheader("📦 Stock Analysis")

total_stock_received = (
    analysis_df["stock_received"].sum()
)

total_stock_demand = (
    analysis_df["demand"].sum()
)

estimated_remaining_stock = (
    total_stock_received -
    total_stock_demand
)

if estimated_remaining_stock < 0:

    stock_status = "⚠️ Potential Stock Shortage"

elif estimated_remaining_stock < average_demand:

    stock_status = "🟡 Low Stock"

else:

    stock_status = "🟢 Stock Level Acceptable"


st.metric(
    "Estimated Remaining Stock",
    f"{estimated_remaining_stock:.0f} units"
)

st.write(
    f"### Status: {stock_status}"
)


# ==========================================================
# REORDER POINT
# ==========================================================

st.subheader("🔄 Reorder Point Analysis")

z_value = st.slider(
    "Service level factor (z)",
    min_value=0.50,
    max_value=3.00,
    value=1.65,
    step=0.05
)

safety_stock = (
    z_value *
    std_demand
)

reorder_point = (
    mean_demand +
    safety_stock
)

col1, col2 = st.columns(2)

col1.metric(
    "Safety Stock",
    f"{safety_stock:.2f} units"
)

col2.metric(
    "Recommended Reorder Point",
    f"{reorder_point:.2f} units"
)


# ==========================================================
# SHORTAGE ALERT
# ==========================================================

st.subheader("🚨 Stock Shortage Alert")

if estimated_remaining_stock < reorder_point:

    st.error(
        """
        Stock level is below the recommended
        reorder point. Restocking should be considered.
        """
    )

else:

    st.success(
        """
        Current estimated stock is above the
        recommended reorder point.
        """
    )


# ==========================================================
# SUMMARY
# ==========================================================

st.subheader("📝 Automatic Summary")

st.write(
    f"""
    - Average medication demand is **{mean_demand:.2f} units per period**.
    - Maximum observed demand is **{maximum_demand} units**.
    - Estimated Poisson parameter λ is **{lambda_value:.2f}**.
    - Estimated safety stock is **{safety_stock:.2f} units**.
    - Recommended reorder point is **{reorder_point:.2f} units**.
    """
)

if high_demand_probability > 0.20:

    st.warning(
        "The probability of a high-demand period is relatively high."
    )

else:

    st.success(
        "The probability of a high-demand period is relatively low."
    )


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "Medication Demand and Stock Analysis | "
    "Probability & Stochastic Process Project"
)
