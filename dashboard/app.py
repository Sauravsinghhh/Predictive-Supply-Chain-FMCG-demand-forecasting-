"""
FreshMind Dashboard - Week 2 End-to-End Demostration.
Provides an interactive Streamlit UI to visualize historical sales,
generate baseline demand forecasts, and compute replenishment order recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys

# Add project root to path for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.data_loader import DataLoader
from src.features.build_features import prepare_single_series, add_rolling_features
from src.models.baseline_forecast import NaiveForecaster, SeasonalNaiveForecaster, MovingAverageForecaster
from src.inventory.replenishment import calculate_replenishment_order

# Page configuration
st.set_page_config(
    page_title="FreshMind - Demand Forecasting & Replenishment Dashboard",
    page_icon="🍏",
    layout="wide"
)

# Custom premium styling
st.markdown("""
<style>
    /* Main container and title */
    .reportview-container {
        background: #0e1117;
    }
    h1, h2, h3 {
        font-family: 'Inter', -apple-system, sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    /* Elegant card containers */
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        margin-bottom: 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #4f46e5;
    }
    .metric-label {
        font-size: 13px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .metric-val {
        font-size: 36px;
        font-weight: 800;
        color: #f8fafc;
        background: linear-gradient(to right, #ffffff, #cbd5e1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .metric-val-colored {
        font-size: 36px;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8 0%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .alert-banner {
        background-color: #7f1d1d;
        border: 1px solid #b91c1c;
        border-radius: 8px;
        padding: 12px 16px;
        color: #fca5a5;
        font-weight: 500;
        margin-top: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Title & Description
st.title("FreshMind Demand Forecasting & Replenishment")
st.markdown("---")

@st.cache_data
def load_all_data():
    """Caches data loading so interactions are near-instant."""
    loader = DataLoader(config_path="configs/config.yaml")
    calendar_df, prices_df, sales_df = loader.load_raw_dataframes()
    return calendar_df, prices_df, sales_df

try:
    calendar_df, prices_df, sales_df = load_all_data()
except Exception as e:
    st.error(f"Error loading datasets: {e}")
    st.info("Make sure the raw data is ingested by running: python scripts/download_data.py --sample")
    st.stop()

# Sidebar panel for parameters and filters
st.sidebar.header("📊 Planning Filters & Inputs")
st.sidebar.markdown("---")

# Store Selection
stores = sorted(sales_df["store_id"].unique())
selected_store = st.sidebar.selectbox("🏬 Select Store", stores, index=0)

# Filter items for selected store
items = sorted(sales_df[sales_df["store_id"] == selected_store]["item_id"].unique())
selected_item = st.sidebar.selectbox("📦 Select SKU", items, index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("🔮 Forecasting Options")

# Baseline selection
model_choice = st.sidebar.selectbox(
    "📐 Baseline Model",
    ["Seasonal Naive", "Naive Forecast", "Simple Moving Average (7-day)", "Simple Moving Average (14-day)"]
)

# Forecast Horizon
horizon = st.sidebar.slider("📅 Forecast Horizon (Days)", min_value=7, max_value=28, value=28, step=7)

st.sidebar.markdown("---")
st.sidebar.subheader("📦 Inventory Policy")

# Inventory parameters
current_inv = st.sidebar.number_input("📥 Current Inventory (Units)", min_value=0.0, value=10.0, step=1.0)
safety_buff = st.sidebar.number_input("🛡️ Safety Buffer (Units)", min_value=0.0, value=15.0, step=1.0)

# Main Dashboard Grid Layout
col1, col2 = st.columns([2, 5])

with col1:
    st.subheader("📋 Context & Metadata")
    # Retrieve product info
    item_rows = sales_df[(sales_df["store_id"] == selected_store) & (sales_df["item_id"] == selected_item)]
    
    if not item_rows.empty:
        dept = item_rows.iloc[0]["dept_id"]
        cat = item_rows.iloc[0]["cat_id"]
        state = item_rows.iloc[0]["state_id"]
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">SKU ID</div>
            <div style="font-size: 18px; font-weight: 600; color: #f8fafc; margin-bottom: 12px;">{selected_item}</div>
            
            <div class="metric-label">Store & State</div>
            <div style="font-size: 16px; font-weight: 500; color: #cbd5e1; margin-bottom: 12px;">{selected_store} ({state})</div>
            
            <div class="metric-label">Category / Department</div>
            <div style="font-size: 16px; font-weight: 500; color: #cbd5e1;">{cat} / {dept}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("SKU details not found.")

# Process single-series data
try:
    single_series_df = prepare_single_series(sales_df, calendar_df, selected_store, selected_item)
    single_series_df = add_rolling_features(single_series_df)
except Exception as e:
    st.error(f"Error preparing time-series data: {e}")
    st.stop()

# Split data: We forecast beyond the end of the history
historical_sales = single_series_df["sales"].values
historical_dates = single_series_df["date"].values

# Select and train baseline forecaster
if model_choice == "Naive Forecast":
    forecaster = NaiveForecaster()
elif model_choice == "Seasonal Naive":
    forecaster = SeasonalNaiveForecaster(period=7)
elif model_choice == "Simple Moving Average (7-day)":
    forecaster = MovingAverageForecaster(window=7)
else:
    forecaster = MovingAverageForecaster(window=14)

forecaster.fit(historical_sales)
predictions = forecaster.predict(horizon=horizon)

# Create future dates
last_date = historical_dates[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon)

# Replenishment calculation
forecast_sum = float(np.sum(predictions))
replenishment_results = calculate_replenishment_order(
    forecast_demand_sum=forecast_sum,
    current_inventory=current_inv,
    safety_buffer=safety_buff
)

with col2:
    st.subheader("💡 Replenishment Decision & Forecast Summary")
    
    # 3-Column Metrics Grid
    m_col1, m_col2, m_col3 = st.columns(3)
    
    with m_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Forecasted Demand ({horizon}d)</div>
            <div class="metric-val">{forecast_sum:.1f}</div>
            <div style="font-size: 12px; color: #94a3b8;">Expected units sold</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Safety Buffer</div>
            <div class="metric-val">{safety_buff:.1f}</div>
            <div style="font-size: 12px; color: #94a3b8;">Minimum stock cushion</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m_col3:
        recommended_qty = replenishment_results["recommended_order"]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Recommended Order Qty</div>
            <div class="metric-val-colored">{recommended_qty:.1f}</div>
            <div style="font-size: 12px; color: #818cf8;">Units to order immediately</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Warning details
    if replenishment_results["under_stock_alert"]:
        st.markdown(f"""
        <div class="alert-banner">
            ⚠️ <b>Stock Warning:</b> Current inventory ({current_inv:.1f} units) is below the safety stock cushion ({safety_buff:.1f} units). Order replenishment suggested.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success("✅ Stock levels are healthy relative to safety cushion.")

# Visualization section
st.subheader("📈 Historical Sales & Demand Forecast Plot")

# Build a clean visual layout using Plotly
fig = go.Figure()

# Plot historical actual sales (limit to last 30 observed days to keep it clear)
plot_history_days = 30
hist_plot_dates = historical_dates[-plot_history_days:]
hist_plot_sales = historical_sales[-plot_history_days:]

fig.add_trace(go.Scatter(
    x=hist_plot_dates,
    y=hist_plot_sales,
    mode='lines+markers',
    name='Historical Sales',
    line=dict(color='#818cf8', width=3),
    marker=dict(size=6, color='#6366f1'),
    hovertemplate="Date: %{x}<br>Sales: %{y}<extra></extra>"
))

# Plot future predictions
fig.add_trace(go.Scatter(
    x=future_dates,
    y=predictions,
    mode='lines+markers',
    name=f'{model_choice} Forecast',
    line=dict(color='#a855f7', width=3, dash='dash'),
    marker=dict(size=6, color='#d946ef'),
    hovertemplate="Date: %{x}<br>Forecast: %{y:.2f}<extra></extra>"
))

# Safety buffer reference line (if applicable)
fig.add_trace(go.Scatter(
    x=list(hist_plot_dates) + list(future_dates),
    y=[safety_buff] * (len(hist_plot_dates) + len(future_dates)),
    mode='lines',
    name='Safety Buffer Threshold',
    line=dict(color='#ef4444', width=1, dash='dot'),
    hovertemplate="Safety Buffer: %{y}<extra></extra>"
))

fig.update_layout(
    plot_bgcolor='#1e293b',
    paper_bgcolor='#0f172a',
    margin=dict(l=40, r=40, t=20, b=40),
    hovermode="x unified",
    xaxis=dict(
        showgrid=True,
        gridcolor='#334155',
        tickfont=dict(color='#94a3b8'),
        title_font=dict(color='#94a3b8')
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor='#334155',
        tickfont=dict(color='#94a3b8'),
        title_font=dict(color='#94a3b8'),
        title="Units Sold / Forecasted"
    ),
    legend=dict(
        font=dict(color='#e2e8f0'),
        bgcolor='rgba(30, 41, 59, 0.8)',
        bordercolor='#334155',
        borderwidth=1
    )
)

st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 12px;'>"
    "FreshMind Supply Chain Analytics Engine © 2026. Built for FreshKart Replenishment Optimization."
    "</div>",
    unsafe_allow_html=True
)
