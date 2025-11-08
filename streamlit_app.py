import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pandas.tseries.offsets import DateOffset
import warnings
from sklearn.metrics import mean_absolute_percentage_error as calculate_mape

# Suppress warnings from statsmodels, which are common in Streamlit environments
warnings.filterwarnings("ignore")

# --- 1. Data Loading and Initial Preprocessing ---
# Data snippet provided by the user from the uploaded file
CSV_CONTENT = """Barangay,Year,Quarter,Period,Copra_Production (MT),Farmgate Price (PHP/kg),Millgate Price (PHP/kg)
Poblacion,2015,Q1,2015-01-01,32.17,24.50,28.50
Poblacion,2015,Q2,2015-04-01,32.15,23.00,27.00
Poblacion,2015,Q3,2015-07-01,32.18,21.00,25.00
Poblacion,2015,Q4,2015-10-01,33.13,16.50,20.50
Poblacion,2016,Q1,2016-01-01,33.00,18.00,22.00
Poblacion,2016,Q2,2016-04-01,33.00,20.50,24.50
Poblacion,2016,Q3,2016-07-01,32.00,23.00,27.00
Poblacion,2016,Q4,2016-10-01,33.18,26.50,30.50
Poblacion,2017,Q1,2017-01-01,32.00,42.00,46.00
Poblacion,2017,Q2,2017-04-01,32.00,45.00,49.00
Poblacion,2017,Q3,2017-07-01,31.00,46.50,50.50
Poblacion,2017,Q4,2017-10-01,33.00,46.00,50.00
Poblacion,2018,Q1,2018-01-01,32.00,38.00,42.00
Poblacion,2018,Q2,2018-04-01,32.00,28.00,32.00
Poblacion,2018,Q3,2018-07-01,33.00,22.00,26.00
Poblacion,2018,Q4,2018-10-01,31.00,18.00,22.00
Poblacion,2019,Q1,2019-01-01,7.17,14.00,18.00
Poblacion,2019,Q2,2019-04-01,7.15,14.50,18.50
Poblacion,2019,Q3,2019-07-01,7.18,15.00,19.00
Poblacion,2019,Q4,2019-10-01,7.12,16.50,20.50
Poblacion,2020,Q1,2020-01-01,7.09,20.00,24.00
Poblacion,2020,Q2,2020-04-01,7.10,24.00,28.00
Poblacion,2020,Q3,2020-07-01,7.02,25.50,29.50
Poblacion,2020,Q4,2020-10-01,7.30,28.50,32.50
Poblacion,2021,Q1,2021-01-01,7.40,31.00,35.00
Poblacion,2021,Q2,2021-04-01,7.10,34.00,38.00
Poblacion,2021,Q3,2021-07-01,7.30,35.50,39.50
Poblacion,2021,Q4,2021-10-01,7.40,36.50,40.50
Poblacion,2022,Q1,2022-01-01,14.58,40.00,44.00
Poblacion,2022,Q2,2022-04-01,14.60,48.00,57.10
Poblacion,2022,Q3,2022-07-01,14.49,36.00,40.00
Poblacion,2022,Q4,2022-10-01,14.57,28.00,32.00
Poblacion,2023,Q1,2023-01-01,35.61,23.85,34.00
Poblacion,2023,Q2,2023-04-01,35.60,25.00,35.00
Poblacion,2023,Q3,2023-07-01,35.70,27.00,37.00
Poblacion,2023,Q4,2023-10-01,35.65,29.00,39.00
Poblacion,2024,Q1,2024-01-01,38.40,32.00,42.00
Poblacion,2024,Q2,2024-04-01,38.10,34.00,44.00
Poblacion,2024,Q3,2024-07-01,38.90,34.07,44.18
Poblacion,2024,Q4,2024-10-01,39.20,40.00,50.00
Poblacion,2025,Q1,2025-01-01,40.42,48.64,52.12
Poblacion,2025,Q2,2025-04-01,40.40,64.55,81.00
Poblacion,2025,Q3,2025-07-01,40.70,56.79,72.70
Bunawan Brook,2015,Q1,2015-01-01,417.45,24.50,28.50
Bunawan Brook,2015,Q2,2015-04-01,415.12,23.00,27.00
Bunawan Brook,2015,Q3,2015-07-01,412.80,21.00,25.00
Bunawan Brook,2015,Q4,2015-10-01,416.21,16.50,20.50
Bunawan Brook,2016,Q1,2016-01-01,419.92,18.00,22.00
Bunawan Brook,2016,Q2,2016-04-01,419.18,20.50,24.50
Bunawan Brook,2016,Q3,2016-07-01,419.71,23.00,27.00
Bunawan Brook,2016,Q4,2016-10-01,420.28,26.50,30.50
Bunawan Brook,2017,Q1,2017-01-01,411.20,42.00,46.00
Bunawan Brook,2017,Q2,2017-04-01,411.30,45.00,49.00
Bunawan Brook,2017,Q3,2017-07-01,411.50,46.50,50.50
Bunawan Brook,2017,Q4,2017-10-01,411.20,46.00,50.00
Bunawan Brook,2018,Q1,2018-01-01,416.00,38.00,42.00
Bunawan Brook,2018,Q2,2018-04-01,415.00,28.00,32.00
Bunawan Brook,2018,Q3,2018-07-01,416.00,22.00,26.00
Bunawan Brook,2018,Q4,2018-10-01,417.00,18.00,22.00
Bunawan Brook,2019,Q1,2019-01-01,89.14,14.00,18.00
Bunawan Brook,2019,Q2,2019-04-01,89.50,14.50,18.50
Bunawan Brook,2019,Q3,2019-07-01,89.90,15.00,19.00
Bunawan Brook,2019,Q4,2019-10-01,89.15,16.50,20.50
Bunawan Brook,2020,Q1,2020-01-01,93.83,20.00,24.00
Bunawan Brook,2020,Q2,2020-04-01,93.80,24.00,28.00
Bunawan Brook,2020,Q3,2020-07-01,93.90,25.50,29.50
Bunawan Brook,2020,Q4,2020-10-01,93.01,28.50,32.50
Bunawan Brook,2021,Q1,2021-01-01,95.80,31.00,35.00
Bunawan Brook,2021,Q2,2021-04-01,95.70,34.00,38.00
Bunawan Brook,2021,Q3,2021-07-01,95.80,35.50,39.50
Bunawan Brook,2021,Q4,2021-10-01,95.60,36.50,40.50
Bunawan Brook,2022,Q1,2022-01-01,178.40,40.00,44.00
Bunawan Brook,2022,Q2,2022-04-01,178.50,48.00,57.10
Bunawan Brook,2022,Q3,2022-07-01,178.60,36.00,40.00
Bunawan Brook,2022,Q4,2022-10-01,178.50,28.00,32.00
Bunawan Brook,2023,Q1,2023-01-01,1147.03,23.85,34.00
Bunawan Brook,2023,Q2,2023-04-01,1147.00,25.00,35.00
Bunawan Brook,2023,Q3,2023-07-01,1148.00,27.00,37.00
Bunawan Brook,2023,Q4,2023-10-01,1147.00,29.00,39.00
Bunawan Brook,2024,Q1,2024-01-01,560.00,32.00,42.00
Bunawan Brook,2024,Q2,2024-04-01,561.00,34.00,44.00
Bunawan Brook,2024,Q3,2024-07-01,560.00,34.07,44.18
Bunawan Brook,2024,Q4,2024-10-01,562.00,40.00,50.00
Bunawan Brook,2025,Q1,2025-01-01,584.92,48.64,52.12
Bunawan Brook,2025,Q2,2025-04-01,585.00,64.55,81.00
Bunawan Brook,2025,Q3,2025-07-01,586.00,56.79,72.70
Consuelo,2015,Q1,2015-01-01,115.50,24.50,28.50
Consuelo,2015,Q2,2015-04-01,115.51,23.00,27.00
Consuelo,2015,Q3,2015-07-01,114.51,21.00,25.00
Consuelo,2015,Q4,2015-10-01,115.98,16.50,20.50
Consuelo,2016,Q1,2016-01-01,118.80,18.00,22.00
Consuelo,2016,Q2,2016-04-01,118.71,20.50,24.50
Consuelo,2016,Q3,2016-07-01,118.25,23.00,27.00
Consuelo,2016,Q4,2016-10-01,118.34,26.50,30.50
Consuelo,2017,Q1,2017-01-01,118.40,42.00,46.00
Consuelo,2017,Q2,2017-04-01,118.30,45.00,49.00
Consuelo,2017,Q3,2017-07-01,118.70,46.50,50.50
Consuelo,2017,Q4,2017-10-01,118.20,46.00,50.00
Consuelo,2018,Q1,2018-01-01,120.80,38.00,42.00
Consuelo,2018,Q2,2018-04-01,120.00,28.00,32.00
Consuelo,2018,Q3,2018-07-01,119.50,22.00,26.00
Consuelo,2018,Q4,2018-10-01,118.00,18.00,22.00
Consuelo,2019,Q1,2019-01-01,26.72,14.00,18.00
Consuelo,2019,Q2,2019-04-01,26.71,14.50,18.50
Consuelo,2019,Q3,2019-07-01,26.80,15.00,19.00
Consuelo,2019,Q4,2019-10-01,26.79,16.50,20.50
Consuelo,2020,Q1,2020-01-01,25.21,20.00,24.00
Consuelo,2020,Q2,2020-04-01,25.15,24.00,28.00
Consuelo,2020,Q3,2020-07-01,25.20,25.50,29.50
Consuelo,2020,Q4,2020-10-01,25.21,28.50,32.50
Consuelo,2021,Q1,2021-01-01,25.84,31.00,35.00
Consuelo,2021,Q2,2021-04-01,25.80,34.00,38.00
Consuelo,2021,Q3,2021-07-01,25.00,35.50,39.50
Consuelo,2021,Q4,2021-10-01,25.90,36.50,40.50
Consuelo,2022,Q1,2022-01-01,48.70,40.00,44.00
Consuelo,2022,Q2,2022-04-01,48.70,48.00,57.10
Consuelo,2022,Q3,2022-07-01,48.90,36.00,40.00
Consuelo,2022,Q4,2022-10-01,48.30,28.00,32.00
Consuelo,2023,Q1,2023-01-01,392.53,23.85,34.00
Consuelo,2023,Q2,2023-04-01,392.00,25.00,35.00
Consuelo,2023,Q3,2023-07-01,390.00,27.00,37.00
Consuelo,2023,Q4,2023-10-01,391.00,29.00,39.00
Consuelo,2024,Q1,2024-01-01,202.40,32.00,42.00
Consuelo,2024,Q2,2024-04-01,202.40,34.00,44.00
Consuelo,2024,Q3,2024-07-01,203.00,34.07,44.18
Consuelo,2024,Q4,2024-10-01,203.00,40.00,50.00
Consuelo,2025,Q1,2025-01-01,228.94,48.64,52.12
Consuelo,2025,Q2,2025-04-01,228.90,64.55,81.00
Consuelo,2025,Q3,2025-07-01,229.00,56.79,72.70
Libertad,2015,Q1,2015-01-01,16.50,24.50,28.50
Libertad,2015,Q2,2015-04-01,16.30,23.00,27.00
Libertad,2015,Q3,2015-07-01,16.10,21.00,25.00
Libertad,2015,Q4,2015-10-01,16.12,16.50,20.50
Libertad,2016,Q1,2016-01-01,16.50,18.00,22.00
Libertad,2016,Q2,2016-04-01,16.30,20.50,24.50
Libertad,2016,Q3,2016-07-01,16.80,23.00,27.00
Libertad,2016,Q4,2016-10-01,16.29,26.50,30.50
Libertad,2017,Q1,2017-01-01,16.00,42.00,46.00
Libertad,2017,Q2,2017-04-01,16.00,45.00,49.00
Libertad,2017,Q3,2017-07-01,15.80,46.50,50.50
Libertad,2017,Q4,2017-10-01,16.30,46.00,50.00
Libertad,2018,Q1,2018-01-01,16.00,38.00,42.00
Libertad,2018,Q2,2018-04-01,16.00,28.00,32.00
Libertad,2018,Q3,2018-07-01,15.00,22.00,26.00
Libertad,2018,Q4,2018-10-01,15.30,18.00,22.00
Libertad,2019,Q1,2019-01-01,3.50,14.00,18.00
Libertad,2019,Q2,2019-04-01,3.30,14.50,18.50
Libertad,2019,Q3,2019-07-01,3.50,15.00,19.00
Libertad,2019,Q4,2019-10-01,3.50,16.50,20.50
Libertad,2020,Q1,2020-01-01,3.23,20.00,24.00
Libertad,2020,Q2,2020-04-01,3.21,24.00,28.00
Libertad,2020,Q3,2020-07-01,3.25,25.50,29.50
Libertad,2020,Q4,2020-10-01,3.25,28.50,32.50
Libertad,2021,Q1,2021-01-01,3.15,31.00,35.00
Libertad,2021,Q2,2021-04-01,3.15,34.00,38.00
Libertad,2021,Q3,2021-07-01,3.15,35.50,39.50
Libertad,2021,Q4,2021-10-01,3.15,36.50,40.50
Libertad,2022,Q1,2022-01-01,5.83,40.00,44.00
Libertad,2022,Q2,2022-04-01,5.81,48.00,57.10
Libertad,2022,Q3,2022-07-01,5.79,36.00,40.00
Libertad,2022,Q4,2022-10-01,5.85,28.00,32.00
Libertad,2023,Q1,2023-01-01,34.16,23.85,34.00
Libertad,2023,Q2,2023-04-01,34.17,25.00,35.00
Libertad,2023,Q3,2023-07-01,34.18,27.00,37.00
Libertad,2023,Q4,2023-10-01,35.10,29.00,39.00
Libertad,2024,Q1,2024-01-01,16.80,32.00,42.00
Libertad,2024,Q2,2024-04-01,16.90,34.00,44.00
Libertad,2024,Q3,2024-07-01,16.70,34.07,44.18
Libertad,2024,Q4,2024-10-01,17.10,40.00,50.00
Libertad,2025,Q1,2025-01-01,17.66,48.64,52.12
Libertad,2025,Q2,2025-04-01,17.60,64.55,81.00
Libertad,2025,Q3,2025-07-01,17.80,56.79,72.70
Imelda,2015,Q1,2015-01-01,74.25,24.50,28.50
Imelda,2015,Q2,2015-04-01,74.29,23.00,27.00
Imelda,2015,Q3,2015-07-01,73.68,21.00,25.00
Imelda,2015,Q4,2015-10-01,74.57,16.50,20.50
Imelda,2016,Q1,2016-01-01,78.37,18.00,22.00
Imelda,2016,Q2,2016-04-01,78.32,20.50,24.50
Imelda,2016,Q3,2016-07-01,79.15,23.00,27.00
Imelda,2016,Q4,2016-10-01,79.36,26.50,30.50
Imelda,2017,Q1,2017-01-01,78.40,42.00,46.00
Imelda,2017,Q2,2017-04-01,78.50,45.00,49.00
Imelda,2017,Q3,2017-07-01,77.90,46.50,50.50
Imelda,2017,Q4,2017-10-01,78.30,46.00,50.00
Imelda,2018,Q1,2018-01-01,80.80,38.00,42.00
Imelda,2018,Q2,2018-04-01,80.30,28.00,32.00
Imelda,2018,Q3,2018-07-01,79.50,22.00,26.00
Imelda,2018,Q4,2018-10-01,79.90,18.00,22.00
Imelda,2019,Q1,2019-01-01,18.10,14.00,18.00
Imelda,2019,Q2,2019-04-01,18.20,14.50,18.50
Imelda,2019,Q3,2019-07-01,18.10,15.00,19.00
Imelda,2019,Q4,2019-10-01,18.50,16.50,20.50
Imelda,2020,Q1,2020-01-01,17.17,20.00,24.00
Imelda,2020,Q2,2020-04-01,17.15,24.00,28.00
Imelda,2020,Q3,2020-07-01,17.21,25.50,29.50
Imelda,2020,Q4,2020-10-01,17.30,28.50,32.50
Imelda,2021,Q1,2021-01-01,17.64,31.00,35.00
Imelda,2021,Q2,2021-04-01,17.60,34.00,38.00
Imelda,2021,Q3,2021-07-01,17.50,35.50,39.50
Imelda,2021,Q4,2021-10-01,17.70,36.50,40.50
Imelda,2022,Q1,2022-01-01,32.95,40.00,44.00
Imelda,2022,Q2,2022-04-01,32.90,48.00,57.10
Imelda,2022,Q3,2022-07-01,32.96,36.00,40.00
Imelda,2022,Q4,2022-10-01,32.89,28.00,32.00
Imelda,2023,Q1,2023-01-01,236.50,23.85,34.00
Imelda,2023,Q2,2023-04-01,236.50,25.00,35.00
Imelda,2023,Q3,2023-07-01,236.70,27.00,37.00
Imelda,2023,Q4,2023-10-01,236.90,29.00,39.00
Imelda,2024,Q1,2024-01-01,115.36,32.00,42.00
Imelda,2024,Q2,2024-04-01,115.50,34.00,44.00
Imelda,2024,Q3,2024-07-01,115.60,34.07,44.18
Imelda,2024,Q4,2024-10-01,116.30,40.00,50.00
Imelda,2025,Q1,2025-01-01,120.65,48.64,52.12
Imelda,2025,Q2,2025-04-01,120.00,64.55,81.00
Imelda,2025,Q3,2025-07-01,121.50,56.79,72.70
Mambalili,2015,Q1,2015-01-01,12.37,24.50,28.50
Mambalili,2015,Q2,2015-04-01,12.31,23.00,27.00
Mambalili,2015,Q3,2015-07-01,12.52,21.00,25.00
Mambalili,2015,Q4,2015-10-01,12.38,16.50,20.50
Mambalili,2016,Q1,2016-01-01,12.37,18.00,22.00
Mambalili,2016,Q2,2016-04-01,11.98,20.50,24.50
Mambalili,2016,Q3,2016-07-01,11.81,23.00,27.00
Mambalili,2016,Q4,2016-10-01,12.68,26.50,30.50
Mambalili,2017,Q1,2017-01-01,12.00,42.00,46.00
Mambalili,2017,Q2,2017-04-01,12.00,45.00,49.00
Mambalili,2017,Q3,2017-07-01,13.00,46.50,50.50
Mambalili,2017,Q4,2017-10-01,12.00,46.00,50.00
Mambalili,2018,Q1,2018-01-01,12.00,38.00,42.00
Mambalili,2018,Q2,2018-04-01,12.10,28.00,32.00
Mambalili,2018,Q3,2018-07-01,12.50,22.00,26.00
Mambalili,2018,Q4,2018-10-01,12.90,18.00,22.00
Mambalili,2019,Q1,2019-01-01,2.86,14.00,18.00
Mambalili,2019,Q2,2019-04-01,2.83,14.50,18.50
Mambalili,2019,Q3,2019-07-01,2.86,15.00,19.00
Mambalili,2019,Q4,2019-10-01,2.70,16.50,20.50
Mambalili,2020,Q1,2020-01-01,2.60,20.00,24.00
Mambalili,2020,Q2,2020-04-01,2.10,24.00,28.00
Mambalili,2020,Q3,2020-07-01,2.30,25.50,29.50
Mambalili,2020,Q4,2020-10-01,2.50,28.50,32.50
Mambalili,2021,Q1,2021-01-01,2.68,31.00,35.00
Mambalili,2021,Q2,2021-04-01,2.70,34.00,38.00
Mambalili,2021,Q3,2021-07-01,2.60,35.50,39.50
Mambalili,2021,Q4,2021-10-01,2.64,36.50,40.50
Mambalili,2022,Q1,2022-01-01,4.96,40.00,44.00
Mambalili,2022,Q2,2022-04-01,4.95,48.00,57.10
Mambalili,2022,Q3,2022-07-01,4.90,36.00,40.00
Mambalili,2022,Q4,2022-10-01,4.70,28.00,32.00
Mambalili,2023,Q1,2023-01-01,27.92,23.85,34.00
Mambalili,2023,Q2,2023-04-01,27.90,25.00,35.00
Mambalili,2023,Q3,2023-07-01,28.20,27.00,37.00
Mambalili,2023,Q4,2023-10-01,27.60,29.00,39.00
Mambalili,2024,Q1,2024-01-01,13.60,32.00,42.00
Mambalili,2024,Q2,2024-04-01,13.50,34.00,44.00
Mambalili,2024,Q3,2024-07-01,13.80,34.07,44.18
Mambalili,2024,Q4,2024-10-01,14.30,40.00,50.00
Mambalili,2025,Q1,2025-01-01,14.29,48.64,52.12
Mambalili,2025,Q2,2025-04-01,14.30,64.55,81.00
Mambalili,2025,Q3,2025-07-01,14.40,56.79,72.70
San Andres,2015,Q1,2015-01-01,74.25,24.50,28.50
San Andres,2015,Q2,2015-04-01,74.21,23.00,27.00
San Andres,2015,Q3,2015-07-01,74.30,21.00,25.00
San Andres,2015,Q4,2015-10-01,75.22,16.50,20.50
San Andres,2016,Q1,2016-01-01,76.72,18.00,22.00
San Andres,2016,Q2,2016-04-01,76.82,20.50,24.50
San Andres,2016,Q3,2016-07-01,77.31,23.00,27.00
San Andres,2016,Q4,2016-10-01,78.10,26.50,30.50
San Andres,2017,Q1,2017-01-01,75.20,42.00,46.00
San Andres,2017,Q2,2017-04-01,75.00,45.00,49.00
San Andres,2017,Q3,2017-07-01,74.00,46.50,50.50
San Andres,2017,Q4,2017-10-01,73.00,46.00,50.00
San Andres,2018,Q1,2018-01-01,75.20,38.00,42.00
San Andres,2018,Q2,2018-04-01,75.10,28.00,32.00
San Andres,2018,Q3,2018-07-01,74.80,22.00,26.00
San Andres,2018,Q4,2018-10-01,74.40,18.00,22.00
San Andres,2019,Q1,2019-01-01,16.43,14.00,18.00
San Andres,2019,Q2,2019-04-01,16.40,14.50,18.50
San Andres,2019,Q3,2019-07-01,16.42,15.00,19.00
San Andres,2019,Q4,2019-10-01,16.47,16.50,20.50
San Andres,2020,Q1,2020-01-01,16.07,20.00,24.00
San Andres,2020,Q2,2020-04-01,16.10,24.00,28.00
San Andres,2020,Q3,2020-07-01,16.05,25.50,29.50
San Andres,2020,Q4,2020-10-01,16.20,28.50,32.50
San Andres,2021,Q1,2021-01-01,17.02,31.00,35.00
San Andres,2021,Q2,2021-04-01,17.01,34.00,38.00
San Andres,2021,Q3,2021-07-01,17.11,35.50,39.50
San Andres,2021,Q4,2021-10-01,17.05,36.50,40.50
San Andres,2022,Q1,2022-01-01,31.78,40.00,44.00
San Andres,2022,Q2,2022-04-01,31.77,48.00,57.10
San Andres,2022,Q3,2022-07-01,31.75,36.00,40.00
San Andres,2022,Q4,2022-10-01,31.80,28.00,32.00
San Andres,2023,Q1,2023-01-01,117.37,23.85,34.00
San Andres,2023,Q2,2023-04-01,117.30,25.00,35.00
San Andres,2023,Q3,2023-07-01,117.50,27.00,37.00
San Andres,2023,Q4,2023-10-01,117.80,29.00,39.00
San Andres,2024,Q1,2024-01-01,86.64,32.00,42.00
San Andres,2024,Q2,2024-04-01,86.60,34.00,44.00
San Andres,2024,Q3,2024-07-01,86.80,34.07,44.18
San Andres,2024,Q4,2024-10-01,86.70,40.00,50.00
San Andres,2025,Q1,2025-01-01,89.80,48.64,52.12
San Andres,2025,Q2,2025-04-01,89.70,64.55,81.00
San Andres,2025,Q3,2025-07-01,89.90,56.79,72.70
San Teodoro,2015,Q1,2015-01-01,33.00,24.50,28.50
San Teodoro,2015,Q2,2015-04-01,34.12,23.00,27.00
San Teodoro,2015,Q3,2015-07-01,33.95,21.00,25.00
San Teodoro,2015,Q4,2015-10-01,34.67,16.50,20.50
San Teodoro,2016,Q1,2016-01-01,34.65,18.00,22.00
San Teodoro,2016,Q2,2016-04-01,33.12,20.50,24.50
San Teodoro,2016,Q3,2016-07-01,32.80,23.00,27.00
San Teodoro,2016,Q4,2016-10-01,33.17,26.50,30.50
San Teodoro,2017,Q1,2017-01-01,33.60,42.00,46.00
San Teodoro,2017,Q2,2017-04-01,33.00,45.00,49.00
San Teodoro,2017,Q3,2017-07-01,34.00,46.50,50.50
San Teodoro,2017,Q4,2017-10-01,32.00,46.00,50.00
San Teodoro,2018,Q1,2018-01-01,33.60,38.00,42.00
San Teodoro,2018,Q2,2018-04-01,33.90,28.00,32.00
San Teodoro,2018,Q3,2018-07-01,33.51,22.00,26.00
San Teodoro,2018,Q4,2018-10-01,32.68,18.00,22.00
San Teodoro,2019,Q1,2019-01-01,7.48,14.00,18.00
San Teodoro,2019,Q2,2019-04-01,7.48,14.50,18.50
San Teodoro,2019,Q3,2019-07-01,7.60,15.00,19.00
San Teodoro,2019,Q4,2019-10-01,7.59,16.50,20.50
San Teodoro,2020,Q1,2020-01-01,6.93,20.00,24.00
San Teodoro,2020,Q2,2020-04-01,6.91,24.00,28.00
San Teodoro,2020,Q3,2020-07-01,6.90,25.50,29.50
San Teodoro,2020,Q4,2020-10-01,6.70,28.50,32.50
San Teodoro,2021,Q1,2021-01-01,7.41,31.00,35.00
San Teodoro,2021,Q2,2021-04-01,7.40,34.00,38.00
San Teodoro,2021,Q3,2021-07-01,7.44,35.50,39.50
San Teodoro,2021,Q4,2021-10-01,7.50,36.50,40.50
San Teodoro,2022,Q1,2022-01-01,13.70,40.00,44.00
San Teodoro,2022,Q2,2022-04-01,13.70,48.00,57.10
San Teodoro,2022,Q3,2022-07-01,13.69,36.00,40.00
San Teodoro,2022,Q4,2022-10-01,13.80,28.00,32.00
San Teodoro,2023,Q1,2023-01-01,80.18,23.85,34.00
San Teodoro,2023,Q2,2023-04-01,80.20,25.00,35.00
San Teodoro,2023,Q3,2023-07-01,80.60,27.00,37.00
San Teodoro,2023,Q4,2023-10-01,80.20,29.00,39.00
San Teodoro,2024,Q1,2024-01-01,39.60,32.00,42.00
San Teodoro,2024,Q2,2024-04-01,39.50,34.00,44.00
San Teodoro,2024,Q3,2024-07-01,39.80,34.07,44.18
San Teodoro,2024,Q4,2024-10-01,40.00,40.00,50.00
San Teodoro,2025,Q1,2025-01-01,41.51,48.64,52.12
San Teodoro,2025,Q2,2025-04-01,41.55,64.55,81.00
San Teodoro,2025,Q3,2025-07-01,41.80,56.79,72.70
Nueva Era,2015,Q1,2015-01-01,11.55,24.50,28.50
Nueva Era,2015,Q2,2015-04-01,11.51,23.00,27.00
Nueva Era,2015,Q3,2015-07-01,11.59,21.00,25.00
Nueva Era,2015,Q4,2015-10-01,11.78,16.50,20.50
Nueva Era,2016,Q1,2016-01-01,11.55,18.00,22.00
Nueva Era,2016,Q2,2016-04-01,11.57,20.50,24.50
Nueva Era,2016,Q3,2016-07-01,11.71,23.00,27.00
Nueva Era,2016,Q4,2016-10-01,12.13,26.50,30.50
Nueva Era,2017,Q1,2017-01-01,11.20,42.00,46.00
Nueva Era,2017,Q2,2017-04-01,11.30,45.00,49.00
Nueva Era,2017,Q3,2017-07-01,11.00,46.50,50.50
Nueva Era,2017,Q4,2017-10-01,11.10,46.00,50.00
Nueva Era,2018,Q1,2018-01-01,11.20,38.00,42.00
Nueva Era,2018,Q2,2018-04-01,11.00,28.00,32.00
Nueva Era,2018,Q3,2018-07-01,11.30,22.00,26.00
Nueva Era,2018,Q4,2018-10-01,12.10,18.00,22.00
Nueva Era,2019,Q1,2019-01-01,2.56,14.00,18.00
Nueva Era,2019,Q2,2019-04-01,2.55,14.50,18.50
Nueva Era,2019,Q3,2019-07-01,2.58,15.00,19.00
Nueva Era,2019,Q4,2019-10-01,2.49,16.50,20.50
Nueva Era,2020,Q1,2020-01-01,2.36,20.00,24.00
Nueva Era,2020,Q2,2020-04-01,2.60,24.00,28.00
Nueva Era,2020,Q3,2020-07-01,2.35,25.50,29.50
Nueva Era,2020,Q4,2020-10-01,2.40,28.50,32.50
Nueva Era,2021,Q1,2021-01-01,2.36,31.00,35.00
Nueva Era,2021,Q2,2021-04-01,2.31,34.00,38.00
Nueva Era,2021,Q3,2021-07-01,2.37,35.50,39.50
Nueva Era,2021,Q4,2021-10-01,2.39,36.50,40.50
Nueva Era,2022,Q1,2022-01-01,4.25,40.00,44.00
Nueva Era,2022,Q2,2022-04-01,4.21,48.00,57.10
Nueva Era,2022,Q3,2022-07-01,4.26,36.00,40.00
Nueva Era,2022,Q4,2022-10-01,4.20,28.00,32.00
Nueva Era,2023,Q1,2023-01-01,24.63,23.85,34.00
Nueva Era,2023,Q2,2023-04-01,24.64,25.00,35.00
Nueva Era,2023,Q3,2023-07-01,24.70,27.00,37.00
Nueva Era,2023,Q4,2023-10-01,24.80,29.00,39.00
Nueva Era,2024,Q1,2024-01-01,12.00,32.00,42.00
Nueva Era,2024,Q2,2024-04-01,12.00,34.00,44.00
Nueva Era,2024,Q3,2024-07-01,13.00,34.07,44.18
Nueva Era,2024,Q4,2024-10-01,12.00,40.00,50.00
Nueva Era,2025,Q1,2025-01-01,12.37,48.64,52.12
Nueva Era,2025,Q2,2025-04-01,12.40,64.55,81.00
Nueva Era,2025,Q3,2025-07-01,12.50,56.79,72.70
"""

@st.cache_data
def load_data():
    """Loads and preprocesses the Copra Production data."""
    df = pd.read_csv(io.StringIO(CSV_CONTENT))

    # Convert 'Period' to datetime objects and set as index
    df['Period'] = pd.to_datetime(df['Period'])
    
    # Handle any potential missing values by filling with the mean of the column
    df.fillna(df.mean(numeric_only=True), inplace=True)
    
    return df

# Load the data once
df_original = load_data()


# --- 2. ARIMA Forecasting Function (Updated for MAPE) ---

@st.cache_data
def arima_forecast(data_series, forecast_end_year):
    """
    Fits an ARIMA model, calculates MAPE on the last 4 historical points,
    and forecasts the series up to the specified year.
    
    Args:
        data_series (pd.Series): The time series data (Copra Production).
        forecast_end_year (int): The last year to forecast to (e.g., 2035).
        
    Returns:
        tuple: (pd.DataFrame, str, str) -> (Combined Data, Model Summary, MAPE String)
    """
    if data_series.empty or len(data_series) < 5:
        return None, "Error: Insufficient data to perform forecasting.", "N/A"
        
    # 1. Backtest Split for MAPE Calculation (using last 4 quarters for testing)
    n_test = 4
    mape_str = "N/A (Not enough data points for validation)"
    
    try:
        # Check if there is enough data to split for a meaningful MAPE calculation
        if len(data_series) > n_test:
            train_data = data_series[:-n_test]
            test_data = data_series[-n_test:]

            # Temporarily fit model on training data for evaluation
            model_train = ARIMA(train_data, order=(1, 1, 0), freq='QS-JAN')
            model_fit_train = model_train.fit()
            
            # Predict the test period
            test_forecast = model_fit_train.get_forecast(steps=n_test)
            test_pred = test_forecast.predicted_mean
            
            # Calculate MAPE
            mape_value = calculate_mape(test_data.values, test_pred.values) * 100
            mape_str = f"{mape_value:.2f}% (on last 4 quarters)"
            
        # 2. Main Forecast: Fit model on ALL available historical data
        model_full = ARIMA(data_series, order=(1, 1, 0), freq='QS-JAN')
        model_fit_full = model_full.fit()
        
        # Determine the start date for forecasting
        start_date = data_series.index[-1] + DateOffset(months=3)
        
        # Create the future date range (Quarterly Start frequency)
        future_dates = pd.date_range(start=start_date, end=f'{forecast_end_year}-10-01', freq='QS')
        
        # Generate the forecast
        forecast = model_fit_full.get_forecast(steps=len(future_dates))
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
        
        return df_combined, model_fit_full.summary(), mape_str
        
    except Exception as e:
        return None, f"ARIMA Model Error: {e}", "N/A"


# --- 3. Page Functions ---

def main_page(df_original):
    """Displays the single-barangay data editor, visualization, and ARIMA forecast."""
    
    st.title(":coconut: Barangay Production Analysis & Forecasting")
    st.markdown("---")

    # Sidebar for Filtering - Done outside to control the main app loop
    
    # Get unique barangays for selection
    barangays = df_original['Barangay'].unique()
    
    # Use session state to manage selected barangay across pages (if needed), 
    # but for simplicity, we use the sidebar selectbox, which is fine for the single-page experience.
    if 'selected_barangay' not in st.session_state:
        st.session_state.selected_barangay = barangays[0]

    st.sidebar.header("Barangay Selection")
    selected_barangay = st.sidebar.selectbox(
        "Select Barangay for Analysis:",
        options=barangays,
        key='barangay_select'
    )
    
    # --- A. Data Viewer and Editor ---
    st.header(f"1. Raw Data Viewer & Editor for {selected_barangay}")
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

    # Convert the edited DataFrame back to a time series for modeling and visualization
    edited_df['Period'] = pd.to_datetime(edited_df['Period'])
    edited_df = edited_df.set_index('Period').sort_index()
    
    ts_production = edited_df['Copra_Production (MT)']
    ts_farmgate = edited_df['Farmgate Price (PHP/kg)']
    ts_millgate = edited_df['Millgate Price (PHP/kg)']
    last_historical_date = ts_production.index.max()

    # --- B. Trend Analysis & Visualization (Production + Prices) ---
    st.header("2. Historical Trends (Production & Prices)")
    st.subheader(f"Historical Data for {selected_barangay}")

    col1, col2 = st.columns(2)

    with col1:
        st.caption("Copra Production (Metric Tons)")
        # Production Line Plot
        fig_prod, ax_prod = plt.subplots(figsize=(10, 5))
        ts_production.plot(ax=ax_prod, marker='o', linestyle='-', color='#0077B6', label='Production (MT)')
        ax_prod.set_title(f'Copra Production Trend')
        ax_prod.set_xlabel('Time (Quarterly)')
        ax_prod.set_ylabel('Production (MT)')
        ax_prod.grid(axis='y', linestyle='--')
        ax_prod.legend(loc='upper left')
        st.pyplot(fig_prod)
        

    with col2:
        st.caption("Farmgate and Millgate Prices (PHP/kg)")
        # Price Line Plot
        fig_price, ax_price = plt.subplots(figsize=(10, 5))
        ts_farmgate.plot(ax=ax_price, marker='s', linestyle='-', color='#48A9A6', label='Farmgate Price')
        ts_millgate.plot(ax=ax_price, marker='^', linestyle='-', color='#F4A261', label='Millgate Price')
        ax_price.set_title(f'Copra Price Trends')
        ax_price.set_xlabel('Time (Quarterly)')
        ax_price.set_ylabel('Price (PHP/kg)')
        ax_price.grid(axis='y', linestyle='--')
        ax_price.legend(loc='upper left')
        st.pyplot(fig_price)

    # --- C. Forecasting ---
    st.header("3. ARIMA Forecasting (2026 - 2035)")
    st.caption(f"Forecasting Copra Production (MT) starting from Q1 of the next period after {last_historical_date.strftime('%Y-%m-%d')}.")

    # Perform the forecast
    df_combined_forecast, model_summary, mape_str = arima_forecast(ts_production, 2035)

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
        

        # --- C2. Forecast Metrics & Table ---
        st.subheader("Forecast Metrics & Data")
        
        st.metric(
            label="ARIMA Model Mean Absolute Percentage Error (MAPE)", 
            value=mape_str,
            help="MAPE is calculated by backtesting the model on the last 4 known historical quarters to estimate predictive accuracy."
        )

        st.markdown("**Forecasted Production Data Table**")
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

def comparison_page(df_original):
    """Displays comparative visualizations for all barangays."""
    
    st.title(":chart_with_upwards_trend: All Barangays Comparison")
    st.markdown("---")
    st.header("1. Production Comparison (Metric Tons)")
    
    # Group and pivot data for plotting all series
    df_pivot_prod = df_original.pivot_table(
        index='Period', 
        columns='Barangay', 
        values='Copra_Production (MT)'
    )
    
    # Plot Production Comparison
    fig_prod, ax_prod = plt.subplots(figsize=(12, 6))
    df_pivot_prod.plot(ax=ax_prod, marker='.', linestyle='-')
    ax_prod.set_title('Copra Production (MT) Comparison Across All Barangays')
    ax_prod.set_xlabel('Period')
    ax_prod.set_ylabel('Copra Production (MT)')
    ax_prod.legend(title='Barangay', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_prod.grid(axis='y', linestyle=':')
    plt.tight_layout()
    st.pyplot(fig_prod)
    
    st.markdown("---")
    
    st.header("2. Price Comparison (Farmgate & Millgate)")
    
    col1, col2 = st.columns(2)
    
    # Plot Farmgate Price Comparison
    with col1:
        df_pivot_farm = df_original.pivot_table(
            index='Period', 
            columns='Barangay', 
            values='Farmgate Price (PHP/kg)'
        )
        fig_farm, ax_farm = plt.subplots(figsize=(10, 5))
        df_pivot_farm.plot(ax=ax_farm, marker='.', linestyle='-')
        ax_farm.set_title('Farmgate Price (PHP/kg) Comparison')
        ax_farm.set_xlabel('Period')
        ax_farm.set_ylabel('Price (PHP/kg)')
        ax_farm.legend(title='Barangay', fontsize=8, loc='upper left')
        ax_farm.grid(axis='y', linestyle=':')
        plt.tight_layout()
        st.pyplot(fig_farm)

    # Plot Millgate Price Comparison
    with col2:
        df_pivot_mill = df_original.pivot_table(
            index='Period', 
            columns='Barangay', 
            values='Millgate Price (PHP/kg)'
        )
        fig_mill, ax_mill = plt.subplots(figsize=(10, 5))
        df_pivot_mill.plot(ax=ax_mill, marker='.', linestyle='-')
        ax_mill.set_title('Millgate Price (PHP/kg) Comparison')
        ax_mill.set_xlabel('Period')
        ax_mill.set_ylabel('Price (PHP/kg)')
        ax_mill.legend(title='Barangay', fontsize=8, loc='upper left')
        ax_mill.grid(axis='y', linestyle=':')
        plt.tight_layout()
        st.pyplot(fig_mill)


# --- 4. Main App Navigation ---

def run_app():
    """Main function to run the Streamlit app with navigation."""
    
    # Setup Streamlit page configuration
    st.set_page_config(layout="wide", page_title="Copra Production & Price Dashboard")
    
    # Load data
    df = load_data()
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a Page",
        ("Barangay Forecast & Analysis", "All Barangays Comparison")
    )
    
    # Display the selected page
    if page == "Barangay Forecast & Analysis":
        main_page(df)
    elif page == "All Barangays Comparison":
        comparison_page(df)

if __name__ == "__main__":
    run_app()
