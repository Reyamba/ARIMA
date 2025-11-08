import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
import warnings
import io

# Set Streamlit page configuration
st.set_page_config(
    page_title="Copra Production Dashboard (ARIMA)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Utility Functions ---

@st.cache_data
def load_data(uploaded_file):
    """Loads and preprocesses the CSV data."""
    try:
        # Use io.StringIO for reading the uploaded CSV file
        df = pd.read_csv(uploaded_file)

        # Standardize column names
        df.columns = [col.strip().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_') for col in df.columns]

        # Convert Period to datetime and set as index
        if 'Period' in df.columns:
            df['Period'] = pd.to_datetime(df['Period'])
            df = df.set_index('Period').sort_index()

        # Rename key columns for simplicity
        if 'Copra_Production_MT' in df.columns:
            df = df.rename(columns={'Copra_Production_MT': 'Production_MT'})
        if 'Barangay' in df.columns:
            df = df.rename(columns={'Barangay': 'Barangay_Name'})

        # Ensure numeric columns are correct type
        numeric_cols = ['Production_MT', 'Farmgate_Price_PHP_kg', 'Millgate_Price_PHP_kg']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df
    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
        return pd.DataFrame()

def run_arima_forecast(df_series, p, d, q, steps):
    """Fits an ARIMA model and generates a forecast."""
    # Suppress warnings from ARIMA convergence
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        try:
            # Fit the ARIMA model
            model = ARIMA(df_series, order=(p, d, q))
            model_fit = model.fit()

            # Forecast future values
            forecast = model_fit.get_forecast(steps=steps)
            forecast_mean = forecast.predicted_mean
            conf_int = forecast.conf_int()

            # Create a results DataFrame
            results_df = pd.DataFrame({
                'Forecasted Production (MT)': forecast_mean.round(2),
                'Lower Bound (95% CI)': conf_int['lower Production_MT'].round(2),
                'Upper Bound (95% CI)': conf_int['upper Production_MT'].round(2)
            })
            results_df.index.name = 'Period'

            return model_fit, results_df, forecast_mean
        except Exception as e:
            st.error(f"ARIMA Model Error: Could not fit the model with order ({p}, {d}, {q}). Please try different parameters. Details: {e}")
            return None, None, None

# --- Main Streamlit App ---

def app():
    st.title("🥥 Copra Production Analysis and Forecasting Dashboard")
    st.markdown("---")

    # Use the uploaded file's internal name
    file_path = "Copra_Production_2015-2025.xlsx - Sheet1.csv"
    uploaded_file = st.uploaded_files[file_path]

    if not uploaded_file:
        st.info("Please ensure the CSV file is available.")
        return

    # Load and process data
    initial_df = load_data(uploaded_file)

    if initial_df.empty:
        return

    st.sidebar.header("Configuration")
    st.sidebar.image("https://placehold.co/100x100/A0522D/FFFFFF?text=Coconut", width=100)
    st.sidebar.markdown("---")

    # --- TAB Structure ---
    tab1, tab2, tab3 = st.tabs(["📊 Data Editor", "📈 Trend Analysis", "🔮 ARIMA Forecast"])

    with tab1:
        st.header("Raw Data Editor")
        st.markdown("You can review and edit the raw data directly in the table below. Note that edits here are temporary for this session.")
        
        # Use st.data_editor to allow editing
        edited_df = st.data_editor(initial_df, use_container_width=True, num_rows="dynamic")
        
        # Update the main DataFrame for use in other tabs
        df = edited_df.copy()
        
        # Display basic stats
        st.subheader("Data Summary")
        st.dataframe(df.describe().T.style.format('{:.2f}'), use_container_width=True)

    # Filter by Barangay (used in both Tab 2 and Tab 3)
    unique_barangays = df['Barangay_Name'].unique()
    selected_barangay = st.sidebar.selectbox(
        "Select Barangay for Analysis:",
        options=unique_barangays
    )

    df_filtered = df[df['Barangay_Name'] == selected_barangay]

    with tab2:
        st.header(f"Trends for {selected_barangay}")

        if df_filtered.empty:
            st.warning(f"No data found for {selected_barangay}.")
        else:
            # 1. Time Series Line Plot (Trend Analysis)
            st.subheader("Copra Production Trend Over Time")
            fig_line = px.line(
                df_filtered.reset_index(),
                x='Period',
                y='Production_MT',
                title=f'Quarterly Copra Production (MT) in {selected_barangay}',
                markers=True,
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_line.update_layout(xaxis_title="Period", yaxis_title="Production (MT)")
            st.plotly_chart(fig_line, use_container_width=True)

            # 2. Annual Bar Chart (Summary Analysis)
            st.subheader("Annual Production Summary")
            annual_df = df_filtered.groupby('Year')['Production_MT'].sum().reset_index()
            fig_bar = px.bar(
                annual_df,
                x='Year',
                y='Production_MT',
                title=f'Total Annual Copra Production (MT) in {selected_barangay}',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_bar.update_layout(xaxis_title="Year", yaxis_title="Total Production (MT)")
            st.plotly_chart(fig_bar, use_container_width=True)


    with tab3:
        st.header(f"ARIMA Forecasting for {selected_barangay}")

        st.markdown("""
        The ARIMA model (AutoRegressive Integrated Moving Average) is used to forecast future values.
        You need to specify the order of the model:
        - **p (AutoRegressive):** The number of lag observations.
        - **d (Integrated):** The number of times the raw observations are differenced.
        - **q (Moving Average):** The size of the moving average window.
        """)

        # --- ARIMA Parameter Inputs ---
        col_p, col_d, col_q, col_steps = st.columns(4)
        with col_p:
            p = st.number_input("P (AR) Order:", min_value=0, value=1, step=1)
        with col_d:
            d = st.number_input("D (Diff) Order:", min_value=0, value=1, step=1)
        with col_q:
            q = st.number_input("Q (MA) Order:", min_value=0, value=1, step=1)
        with col_steps:
            steps = st.number_input("Forecast Steps (Quarters):", min_value=1, value=4, step=1)

        if st.button("Run ARIMA Forecast", type="primary"):
            if df_filtered.empty:
                st.warning("Please ensure data is loaded and a barangay is selected.")
            else:
                # Prepare data for ARIMA
                ts_data = df_filtered['Production_MT'].asfreq('QS').fillna(method='ffill') # 'QS' for Quarter Start
                
                # Run the forecast
                model_fit, forecast_table, forecast_mean = run_arima_forecast(ts_data, p, d, q, steps)

                if forecast_table is not None:
                    st.subheader("Forecasted Copra Production Table")
                    st.dataframe(forecast_table, use_container_width=True)

                    # --- Visualization of Forecast ---
                    st.subheader("Historical Data and Forecast")
                    
                    # Combine historical and forecasted data for plotting
                    history_df = ts_data.to_frame(name='Production_MT')
                    history_df['Type'] = 'Historical'
                    
                    forecast_df = forecast_mean.to_frame(name='Production_MT')
                    forecast_df['Type'] = 'Forecast'
                    
                    # Add CI for historical plot continuity (will only show for forecast, but helps figure setup)
                    history_df['Lower Bound (95% CI)'] = history_df['Production_MT']
                    history_df['Upper Bound (95% CI)'] = history_df['Production_MT']

                    # Prepare final plot data including CI
                    plot_data = pd.concat([history_df, forecast_df])
                    plot_data['Lower Bound (95% CI)'] = pd.concat([history_df['Lower Bound (95% CI)'], forecast_table['Lower Bound (95% CI)']])
                    plot_data['Upper Bound (95% CI)'] = pd.concat([history_df['Upper Bound (95% CI)'], forecast_table['Upper Bound (95% CI)']])


                    fig_forecast = px.line(
                        plot_data.reset_index(),
                        x='Period',
                        y='Production_MT',
                        color='Type',
                        title=f'Copra Production Forecast for {selected_barangay}',
                        markers=True,
                        color_discrete_map={'Historical': '#1f77b4', 'Forecast': '#ff7f0e'}
                    )
                    
                    # Add confidence interval shading
                    fig_forecast.add_traces([
                        px.line(
                            plot_data.reset_index(),
                            x='Period',
                            y='Upper Bound (95% CI)',
                            line_color='rgba(0,0,0,0)',
                            labels={'y': 'Upper Bound (95% CI)'}
                        ).data[0],
                        px.line(
                            plot_data.reset_index(),
                            x='Period',
                            y='Lower Bound (95% CI)',
                            fill='tonexty',
                            fillcolor='rgba(255,127,14,0.2)',
                            line_color='rgba(0,0,0,0)',
                            labels={'y': 'Lower Bound (95% CI)'}
                        ).data[0]
                    ])
                    
                    fig_forecast.update_layout(showlegend=True, yaxis_title="Production (MT)", xaxis_title="Period")
                    st.plotly_chart(fig_forecast, use_container_width=True)


# Execute the app function
if __name__ == '__main__':
    # Streamlit requires a file uploader, but since the file is pre-loaded in this environment, 
    # we bypass the standard uploader and use the internal file structure.
    try:
        app()
    except Exception as e:
        st.error(f"An unexpected error occurred during execution: {e}")
