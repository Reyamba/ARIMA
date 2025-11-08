# PIP_PACKAGES: pandas, numpy, streamlit, plotly, scikit-learn, statsmodels
import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error

# --- Configuration ---
# The filename provided by the user (must match the key in __file_contents)
FILE_NAME = "Copra_Production_2015-2025.xlsx - Sheet1.csv"

# --- Utility Functions ---

def calculate_mape(y_true, y_pred):
    """Calculates Mean Absolute Percentage Error (MAPE)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero by replacing 0 with a small epsilon
    epsilon = 1e-10
    y_true[y_true == 0] = epsilon
    
    # Ensure all inputs are positive for MAPE
    y_true = np.maximum(y_true, epsilon)
    y_pred = np.maximum(y_pred, 0)
    
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def load_data(file_name):
    """Loads and preprocesses the data from the global file content."""
    try:
        # Access the uploaded file content via the global __file_contents dictionary
        if '__file_contents' not in globals() or file_name not in globals()['__file_contents']:
            st.error(f"Data file '{file_name}' not found. Please ensure the file is correctly accessible.")
            return None

        csv_data = globals()['__file_contents'][file_name]
        df = pd.read_csv(io.StringIO(csv_data))

        # Data Cleaning and Preparation
        df.columns = df.columns.str.strip()
        df['Period'] = pd.to_datetime(df['Period'], errors='coerce')
        df['Copra_Production (MT)'] = pd.to_numeric(df['Copra_Production (MT)'], errors='coerce')
        df = df.dropna(subset=['Period', 'Copra_Production (MT)'])
        df['Barangay'] = df['Barangay'].astype('category')

        # Drop duplicates based on (Barangay, Period) and sort
        df = df.sort_values(by=['Barangay', 'Period'])
        df = df.drop_duplicates(subset=['Barangay', 'Period'], keep='last')

        return df

    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
        return None

# --- ARIMA Model and Analysis Function ---

def run_analysis(df_barangay, barangay_name):
    """Performs ARIMA analysis, validation, and forecasting for a single barangay."""
    
    # 1. Prepare Time Series Data
    ts = df_barangay.set_index('Period')['Copra_Production (MT)']

    # 2. Define Training, Validation, and Forecast Periods
    # Use the last 4 quarters (1 year) for validation, and forecast the next 4 (2025).
    
    if len(ts) < 8:
        st.warning(f"Not enough data points for {barangay_name}. Need at least 8 quarters for training (4) and validation (4).")
        return

    # Set validation period dynamically (last 4 quarters)
    validation_start_idx = len(ts) - 4
    
    train_data = ts.iloc[:validation_start_idx]
    validation_data = ts.iloc[validation_start_idx:]

    # Define future forecast dates (the next 4 quarters after the last date)
    last_date = ts.index[-1]
    forecast_dates = pd.date_range(start=last_date, periods=5, freq='QS')[1:] # 'QS' = Quarterly Start

    # 3. Train ARIMA Model
    # Using a common (1, 1, 1) ARIMA model.
    with st.spinner(f"Fitting ARIMA(1, 1, 1) model for {barangay_name}..."):
        try:
            # Setting 'QS' frequency helps the ARIMA model handle the quarterly steps
            model = ARIMA(train_data, order=(1, 1, 1), freq='QS')
            model_fit = model.fit()
        except Exception as e:
            st.error(f"Could not fit ARIMA model. Please check the data for stationarity or missing values. Error: {e}")
            return

    # 4. Predict Validation Data
    try:
        validation_predictions = model_fit.predict(
            start=validation_data.index[0],
            end=validation_data.index[-1],
            typ='levels'
        )
    except Exception as e:
        st.error(f"Error predicting validation data: {e}")
        validation_predictions = pd.Series(np.nan, index=validation_data.index)

    # 5. Forecast Future Data (2025)
    try:
        forecast_results = model_fit.get_forecast(steps=len(forecast_dates))
        forecast = forecast_results.predicted_mean
        forecast.index = forecast_dates
        conf_int = forecast_results.conf_int()
        conf_int.index = forecast_dates
    except Exception as e:
        st.error(f"Error generating future forecast: {e}")
        forecast = pd.Series(np.nan, index=forecast_dates)
        conf_int = pd.DataFrame(np.nan, index=forecast_dates, columns=['lower Copra_Production (MT)', 'upper Copra_Production (MT)'])

    # 6. Calculate Accuracy (MAPE)
    mape = np.nan
    try:
        if not validation_predictions.isna().all():
            valid_y_true = validation_data.values
            valid_y_pred = validation_predictions.values
            mape = calculate_mape(valid_y_true, valid_y_pred)
    except Exception as e:
        st.warning(f"MAPE calculation failed: {e}")

    # --- Display Results ---

    st.markdown("### Model Performance (Accuracy Testing)")
    if not np.isnan(mape):
        st.metric(label="Mean Absolute Percentage Error (MAPE) on Validation Set (Last 4 Quarters)",
                  value=f"{mape:.2f}%",
                  help="Measures the prediction accuracy of the model on the historical validation data. Lower is better.")
    else:
        st.warning("MAPE calculation failed or no valid predictions were made.")

    # --- Trend Analysis and Forecast (Line Plot) ---
    st.markdown("### Trend Analysis and Forecast (Line Plot)")

    fig = go.Figure()

    # Historical Data
    fig.add_trace(go.Scatter(
        x=ts.index,
        y=ts.values,
        mode='lines+markers',
        name='Historical Production',
        line=dict(color='darkblue', width=2)
    ))

    # Forecast Data (2025)
    fig.add_trace(go.Scatter(
        x=forecast.index,
        y=np.maximum(forecast.values, 0), # Ensure non-negative forecast for visualization
        mode='lines+markers',
        name='Forecast',
        line=dict(color='red', dash='dash', width=2)
    ))

    # Confidence Interval
    if not conf_int.empty and not conf_int.isna().all().all():
        fig.add_trace(go.Scatter(
            x=conf_int.index.tolist() + conf_int.index.tolist()[::-1],
            y=conf_int.iloc[:, 1].tolist() + conf_int.iloc[:, 0].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.1)',
            line=dict(color='rgba(255, 255, 255, 0)'),
            name='95% Confidence Interval',
            hoverinfo='skip'
        ))

    fig.update_layout(
        title=f"Copra Production Trend and ARIMA Forecast for {barangay_name}",
        xaxis_title="Period (Quarter)",
        yaxis_title="Copra Production (Metric Tons)",
        hovermode="x unified",
        template="plotly_white",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Model Validation (Bar Chart) ---
    st.markdown("### Model Validation: Actual vs. Predicted (Bar Chart)")

    validation_df = pd.DataFrame({
        'Period': validation_data.index,
        'Actual': validation_data.values,
        'Predicted': validation_predictions.values
    }).set_index('Period')

    # Ensure predictions are non-negative for display
    validation_df['Predicted'] = np.maximum(validation_df['Predicted'], 0)

    fig_bar = px.bar(
        validation_df.reset_index().melt(id_vars='Period', var_name='Type', value_name='Production (MT)'),
        x='Period',
        y='Production (MT)',
        color='Type',
        barmode='group',
        text_auto='.2s',
        title=f'Model Validation: Actual vs. Predicted Copra Production',
        labels={'Production (MT)': 'Copra Production (MT)', 'Period': 'Period'},
        color_discrete_map={'Actual': 'teal', 'Predicted': 'orange'}
    )
    fig_bar.update_layout(height=450)
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- Forecasted Data (Table) ---
    st.markdown("### Forecasted Data (Table)")

    forecast_table = pd.DataFrame({
        'Period': forecast.index.strftime('%Y-Q%q'),
        'Forecasted Production (MT)': np.maximum(forecast.values.round(2), 0) # Ensure non-negative
    })

    if not conf_int.empty and not conf_int.isna().all().all():
        forecast_table['Lower 95% CI'] = np.maximum(conf_int.iloc[:, 0].values.round(2), 0)
        forecast_table['Upper 95% CI'] = np.maximum(conf_int.iloc[:, 1].values.round(2), 0)

    st.dataframe(forecast_table, hide_index=True, use_container_width=True)


# --- Streamlit Main App ---

def app():
    st.set_page_config(layout="wide", page_title="Copra Production ARIMA Analysis")

    st.title("Copra Production Forecasting Dashboard (ARIMA Model)")
    st.markdown("""
        Use the sidebar to select a **Barangay** to view its historical trend, forecasted production (using ARIMA), 
        and model validation results (MAPE and bar chart).
        ***
    """)

    # 1. Load Data
    df = load_data(FILE_NAME)

    if df is not None:
        # Display raw data for editing context
        with st.expander("View Raw Data (for inspection/editing context)"):
            st.dataframe(df, use_container_width=True)
            st.info("Note: For persistent data editing, please modify the source CSV file and re-upload.")

        # 2. Sidebar for Selection
        barangays = sorted(df['Barangay'].unique().tolist())
        st.sidebar.title("Configuration")
        selected_barangay = st.sidebar.selectbox(
            "Select Barangay for Analysis",
            barangays
        )

        # 3. Filter and Run Analysis
        if selected_barangay:
            st.header(f"Results for: {selected_barangay}")
            df_barangay = df[df['Barangay'] == selected_barangay]
            run_analysis(df_barangay, selected_barangay)
        else:
            st.info("Please select a Barangay from the sidebar to begin the trend analysis and forecasting.")

if __name__ == '__main__':
    app()
