pip install -r requirements.txt
import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pandas.tseries.offsets import DateOffset
import warnings

# --- 1. Data Loading and Initial Preprocessing ---
# Load the data from the provided CSV file content
# This ensures the app is fully self-contained and runnable
def load_data():
    """Loads and preprocesses the Copra Production data."""
    # Data snippet provided by the user from the uploaded file
    csv_content = """Barangay,Year,Quarter,Period,Copra_Production (MT),Farmgate Price (PHP/kg),Millgate Price (PHP/kg)
Poblacion,2015,Q1,2015-01-01,32.17,24.5,28.5
Poblacion,2015,Q2,2015-04-01,32.15,23,27
Poblacion,2015,Q3,2015-07-01,32.18,21,25
Poblacion,2015,Q4,2015-10-01,33.13,16.5,20.5
Poblacion,2016,Q1,2016-01-01,33,18,22
Poblacion,2016,Q2,2016-04-01,33,20.5,24.5
Poblacion,2016,Q3,2016-07-01,32,23,27
Poblacion,2016,Q4,2016-10-01,33.18,26.5,30.5
Poblacion,2017,Q1,2017-01-01,32,42,46
Poblacion,2017,Q2,2017-04-01,32,45,49
Poblacion,2017,Q3,2017-07-01,32,46.5,50.5
Poblacion,2017,Q4,2017-10-01,32,46,50
Poblacion,2018,Q1,2018-01-01,32,38,42
Poblacion,2018,Q2,2018-04-01,33,28,32
Poblacion,2018,Q3,2018-07-01,33,22,26
Poblacion,2018,Q4,2018-10-01,33,18,22
Poblacion,2019,Q1,2019-01-01,32.5,14,18
Poblacion,2019,Q2,2019-04-01,32.8,14.5,18.5
Poblacion,2019,Q3,2019-07-01,33.1,13.5,17.5
Poblacion,2019,Q4,2019-10-01,33.2,12.5,16.5
Poblacion,2020,Q1,2020-01-01,33,12,16
Poblacion,2020,Q2,2020-04-01,33,11.5,15.5
Poblacion,2020,Q3,2020-07-01,32.9,11,15
Poblacion,2020,Q4,2020-10-01,33.1,10.5,14.5
Poblacion,2021,Q1,2021-01-01,33.2,11,15
Poblacion,2021,Q2,2021-04-01,33.1,11.5,15.5
Poblacion,2021,Q3,2021-07-01,33.2,12,16
Poblacion,2021,Q4,2021-10-01,33.3,12.5,16.5
Poblacion,2022,Q1,2022-01-01,33.4,13,17
Poblacion,2022,Q2,2022-04-01,33.3,13.5,17.5
Poblacion,2022,Q3,2022-07-01,33.4,14,18
Poblacion,2022,Q4,2022-10-01,33.5,14.5,18.5
Poblacion,2023,Q1,2023-01-01,33.6,15,19
Poblacion,2023,Q2,2023-04-01,33.5,15.5,19.5
Poblacion,2023,Q3,2023-07-01,33.6,16,20
Poblacion,2023,Q4,2023-10-01,33.7,16.5,20.5
Poblacion,2024,Q1,2024-01-01,33.8,17,21
Poblacion,2024,Q2,2024-04-01,33.9,17.5,21.5
Poblacion,2024,Q3,2024-07-01,34.0,18,22
Poblacion,2024,Q4,2024-10-01,34.1,18.5,22.5
Poblacion,2025,Q1,2025-01-01,34.2,19,23
Poblacion,2025,Q2,2025-04-01,34.3,19.5,23.5
Poblacion,2025,Q3,2025-07-01,34.4,20,24
Poblacion,2025,Q4,2025-10-01,34.5,20.5,24.5
Nueva Era,2015,Q1,2015-01-01,10.01,24.5,28.5
Nueva Era,2015,Q2,2015-04-01,10.15,23,27
Nueva Era,2015,Q3,2015-07-01,10.18,21,25
Nueva Era,2015,Q4,2015-10-01,11.13,16.5,20.5
Nueva Era,2016,Q1,2016-01-01,11,18,22
Nueva Era,2016,Q2,2016-04-01,11.2,20.5,24.5
Nueva Era,2016,Q3,2016-07-01,11.71,23,27
Nueva Era,2016,Q4,2016-10-01,12.13,26.5,30.5
Nueva Era,2017,Q1,2017-01-01,11.2,42,46
Nueva Era,2017,Q2,2017-04-01,11.3,45,49
Nueva Era,2017,Q3,2017-07-01,11,46.5,50.5
Nueva Era,2017,Q4,2017-10-01,11.1,46,50
Nueva Era,2018,Q1,2018-01-01,11.2,38,42
Nueva Era,2018,Q2,2018-04-01,11,28,32
Nueva Era,2018,Q3,2018-07-01,11.3,22,26
Nueva Era,2018,Q4,2018-10-01,12.1,18,22
Nueva Era,2019,Q1,2019-01-01,2.56,14,18
Nueva Era,2019,Q2,2019-04-01,2.55,14.5,18.5
Nueva Era,2019,Q3,2019-07-01,2.54,13.5,17.5
Nueva Era,2019,Q4,2019-10-01,2.53,12.5,16.5
Nueva Era,2020,Q1,2020-01-01,2.52,12,16
Nueva Era,2020,Q2,2020-04-01,2.51,11.5,15.5
Nueva Era,2020,Q3,2020-07-01,2.5,11,15
Nueva Era,2020,Q4,2020-10-01,2.49,10.5,14.5
Nueva Era,2021,Q1,2021-01-01,2.48,11,15
Nueva Era,2021,Q2,2021-04-01,2.47,11.5,15.5
Nueva Era,2021,Q3,2021-07-01,2.46,12,16
Nueva Era,2021,Q4,2021-10-01,2.45,12.5,16.5
Nueva Era,2022,Q1,2022-01-01,2.44,13,17
Nueva Era,2022,Q2,2022-04-01,2.43,13.5,17.5
Nueva Era,2022,Q3,2022-07-01,2.42,14,18
Nueva Era,2022,Q4,2022-10-01,2.41,14.5,18.5
Nueva Era,2023,Q1,2023-01-01,2.4,15,19
Nueva Era,2023,Q2,2023-04-01,2.39,15.5,19.5
Nueva Era,2023,Q3,2023-07-01,2.38,16,20
Nueva Era,2023,Q4,2023-10-01,2.37,16.5,20.5
Nueva Era,2024,Q1,2024-01-01,2.36,17,21
Nueva Era,2024,Q2,2024-04-01,2.35,17.5,21.5
Nueva Era,2024,Q3,2024-07-01,2.34,18,22
Nueva Era,2024,Q4,2024-10-01,2.33,18.5,22.5
Nueva Era,2025,Q1,2025-01-01,2.32,19,23
Nueva Era,2025,Q2,2025-04-01,2.31,19.5,23.5
Nueva Era,2025,Q3,2025-07-01,2.3,20,24
Nueva Era,2025,Q4,2025-10-01,2.29,20.5,24.5
"""
    
    df = pd.read_csv(io.StringIO(csv_content))
    
    # Convert 'Period' to datetime objects and set as index
    df['Period'] = pd.to_datetime(df['Period'])
    
    # Handle any potential missing values by filling with the mean of the column
    df.fillna(df.mean(numeric_only=True), inplace=True) 
    
    return df

df_original = load_data()

# --- 2. ARIMA Forecasting Function ---

@st.cache_data
def arima_forecast(data_series, forecast_end_year):
    """
    Fits an ARIMA model and forecasts the series up to the specified year.
    
    Args:
        data_series (pd.Series): The time series data (Copra Production).
        forecast_end_year (int): The last year to forecast to (e.g., 2035).
        
    Returns:
        pd.DataFrame: DataFrame containing historical and forecasted data.
    """
    if data_series.empty or len(data_series) < 5:
        return None, "Error: Insufficient data to perform forecasting."
        
    # Standard practice: Use ARIMA(1, 1, 0) as a robust, simple model for demonstration
    # In a real-world scenario, the (p, d, q) parameters would be determined
    # using auto_arima or ACF/PACF plots.
    try:
        # Check for non-stationarity and apply differencing if needed
        # We assume d=1 (one difference) for most non-stationary time series
        model = ARIMA(data_series, order=(1, 1, 0), freq='QS-JAN')
        model_fit = model.fit()
        
        # Determine the start date for forecasting (the period after the last known data point)
        start_date = data_series.index[-1] + DateOffset(months=3)
        
        # Create the future date range (Quarterly Start frequency)
        future_dates = pd.date_range(start=start_date, end=f'{forecast_end_year}-10-01', freq='QS')
        
        # Generate the forecast
        forecast = model_fit.get_forecast(steps=len(future_dates))
        forecast_values = forecast.predicted_mean
        
        # Create a DataFrame for the forecast results
        df_forecast = pd.DataFrame({
            'Period': future_dates, 
            'Copra_Production (MT)': forecast_values.values,
            'Type': 'Forecast'
        }).set_index('Period')
        
        # Prepare historical data for plotting
        df_historical = pd.DataFrame({
            'Period': data_series.index,
            'Copra_Production (MT)': data_series.values,
            'Type': 'Historical'
        }).set_index('Period')
        
        # Combine historical and forecasted data
        df_combined = pd.concat([df_historical, df_forecast])
        
        return df_combined, model_fit.summary()
        
    except Exception as e:
        return None, f"ARIMA Model Error: {e}"

# --- 3. Streamlit UI and Logic ---

st.set_page_config(layout="wide", page_title="Copra Production & Price Dashboard")

st.title(":coconut: Copra Production Trend Analysis & Forecasting")
st.markdown("---")

# --- Sidebar for Filtering ---
st.sidebar.header("Filter & Model Options")

# Get unique barangays for selection
barangays = df_original['Barangay'].unique()
selected_barangay = st.sidebar.selectbox(
    "Select Barangay for Analysis:",
    options=barangays
)

# --- Main Dashboard Sections ---

# --- A. Data Viewer and Editor ---
st.header("1. Raw Data Viewer & Editor")
st.info("You can directly edit the data below. The model will use the latest values displayed here for forecasting.")

# Filter data for the selected barangay
df_barangay = df_original[df_original['Barangay'] == selected_barangay].reset_index(drop=True)
df_barangay_editable = df_barangay.sort_values(by='Period', ascending=True).copy()

# Use st.data_editor for interactive editing
edited_df = st.data_editor(
    df_barangay_editable,
    column_config={
        "Period": st.column_config.DatetimeColumn("Period", format="YYYY-MM-DD", disabled=True),
        "Barangay": st.column_config.TextColumn("Barangay", disabled=True),
    },
    hide_index=True,
    num_rows="dynamic"
)

# Convert the edited DataFrame back to a time series for modeling
edited_df['Period'] = pd.to_datetime(edited_df['Period'])
ts_data = edited_df.set_index('Period')['Copra_Production (MT)'].sort_index(freq='QS-JAN')
last_historical_date = ts_data.index.max()

# --- B. Trend Analysis & Visualization ---
st.header(f"2. Historical Trends for {selected_barangay}")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Line Plot (Production over Time)")
    # Create line plot
    fig_line, ax_line = plt.subplots(figsize=(10, 5))
    ts_data.plot(ax=ax_line, marker='o', linestyle='-', color='#0077B6')
    ax_line.set_title(f'Copra Production (MT) over Time in {selected_barangay}')
    ax_line.set_xlabel('Time (Quarterly)')
    ax_line.set_ylabel('Copra Production (MT)')
    ax_line.grid(axis='y', linestyle='--')
    st.pyplot(fig_line)
    

with col2:
    st.subheader("Bar Chart (Production Volume)")
    # Create bar chart
    fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
    ts_data.plot(kind='bar', ax=ax_bar, color='#48A9A6')
    
    # Show only year labels on X-axis for readability
    n_ticks = len(ts_data)
    if n_ticks > 0:
        skip_count = max(1, n_ticks // 8) # Show max 8 ticks
        tick_labels = [label.strftime('%Y') if i % skip_count == 0 else '' for i, label in enumerate(ts_data.index)]
        ax_bar.set_xticklabels(tick_labels, rotation=45, ha='right')
        
    ax_bar.set_title(f'Quarterly Copra Production Volume in {selected_barangay}')
    ax_bar.set_xlabel('Period')
    ax_bar.set_ylabel('Copra Production (MT)')
    st.pyplot(fig_bar)
    

# --- C. Forecasting ---
st.header("3. ARIMA Forecasting (2026 - 2035)")
st.caption(f"Forecasting Copra Production (MT) starting from Q1 of the next period after {last_historical_date.strftime('%Y-%m-%d')}.")

# Perform the forecast
# Suppress the warnings from statsmodels that often appear with small datasets
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    df_combined_forecast, model_summary = arima_forecast(ts_data, 2035)

if df_combined_forecast is not None:
    
    # --- C1. Forecast Visualization ---
    st.subheader("Forecast Visualization (Historical + Predicted)")
    
    # Line plot with forecast
    fig_forecast, ax_forecast = plt.subplots(figsize=(12, 6))
    
    # Plot Historical data
    df_combined_forecast[df_combined_forecast['Type'] == 'Historical']['Copra_Production (MT)'].plot(
        ax=ax_forecast, label='Historical Production', color='#1E88E5', linestyle='-'
    )
    
    # Plot Forecast data
    df_combined_forecast[df_combined_forecast['Type'] == 'Forecast']['Copra_Production (MT)'].plot(
        ax=ax_forecast, label='ARIMA Forecast (2035)', color='#FF7043', linestyle='--'
    )
    
    ax_forecast.set_title(f'Copra Production Forecast for {selected_barangay} (2015-2035)')
    ax_forecast.set_xlabel('Period')
    ax_forecast.set_ylabel('Copra Production (MT)')
    ax_forecast.legend()
    ax_forecast.grid(axis='y', linestyle=':')
    
    # Highlight the split point between history and forecast
    ax_forecast.axvline(x=last_historical_date, color='grey', linestyle=':', linewidth=2, label='Forecast Start')
    
    st.pyplot(fig_forecast)
    

    # --- C2. Forecast Table ---
    st.subheader("Forecasted Data Table")
    
    df_table = df_combined_forecast[df_combined_forecast['Type'] == 'Forecast'].copy()
    df_table.index.name = 'Forecast Period'
    df_table['Year'] = df_table.index.year
    df_table['Quarter'] = df_table.index.quarter.map({1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'})
    df_table['Copra_Production (MT)'] = df_table['Copra_Production (MT)'].round(2)
    
    # Final table display
    st.dataframe(
        df_table[['Year', 'Quarter', 'Copra_Production (MT)']],
        height=300
    )

    # --- C3. Model Diagnostics (Optional) ---
    with st.expander("View ARIMA Model Summary"):
        st.code(model_summary)
        st.caption("Note: The model is a simple ARIMA(1, 1, 0) for demonstration purposes. Results may vary.")

else:
    st.error(model_summary)
    st.warning("Please ensure your dataset contains enough clean data points for the selected barangay to run the ARIMA model.")

st.markdown("---")
st.sidebar.markdown("Built with Streamlit & `statsmodels`")
