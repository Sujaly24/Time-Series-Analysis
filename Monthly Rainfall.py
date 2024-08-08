# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 22:43:16 2024

@author: Sujal Yadav
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pmdarima as pm
import statsmodels.api as sm

# Load data
data = pd.read_csv("C:/Users/Sujal Yadav/Desktop/Project/Monthly Rainfall.csv", parse_dates=['Month'], index_col='Month')
data.index = pd.date_range(start='Jan-1901', periods=len(data), freq='ME')

# Plot the data
plt.figure(figsize=(18, 10))
plt.plot(data.index, data)
plt.title('Monthly Rainfall in Delhi')
plt.xlabel('Time')
plt.ylabel('Rainfall')
plt.show()

# Decomposition
decomposition_additive = seasonal_decompose(data, model='additive')
decomposition_additive.plot()
plt.show()


# ADF test
result = adfuller(data.dropna())
print('\nADF Statistic:', result[0])
print('p-value:', result[1])

# ACF and PACF of residuals for Auto ARIMA
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(data, lags=60, ax=plt.gca())
plt.title('ACF Plot')

plt.subplot(122)
plot_pacf(data, lags=60, ax=plt.gca())
plt.title('PACF Plot')

plt.tight_layout()
plt.show()

# Differencing
seasonal_diff = data.diff(12).dropna()

# ADF test on differenced data
result = adfuller(seasonal_diff)
print('ADF Statistic:', result[0])
print('p-value:', result[1])

'''
# Define model parameters
p = [1, 2, 3]
q = [1, 2, 3]
P = [1, 2, 3]
D = [0, 1]
count = 1

# Initialize list to store results
results_list = []

# Loop through model parameters
for i in p:
    for j in q:
        for k in P:
            for l in D:
                print("\nmodel ", count)
                model = SARIMAX(data, order=(i, 0, j), seasonal_order=(k, l, 1, 12))
                results = model.fit()
                
                # Extract metrics from results
                aic = results.aic
                bic = results.bic
                hqic = results.hqic
                log_likelihood = results.llf
                
                # Calculate residuals and Ljung-Box test
                residuals = results.resid
                ljung_box_result = acorr_ljungbox(residuals, lags=[60], return_df=True)
                ljung_box_p_value = ljung_box_result['lb_pvalue'].values[0]
                
                # Calculate MAE, MSE, RMSE
                mae = mean_absolute_error(data, results.fittedvalues)
                mse = mean_squared_error(data, results.fittedvalues)
                rmse = np.sqrt(mse)
                
                # Append metrics to results list
                results_list.append({
                   'Model Number': count,
                   'Order': f'({i}, 0, {j})',
                   'Seasonal Order': f'({k}, {l}, 1, 12)',
                   'AIC': aic,
                   'BIC': bic,
                   'HQIC': hqic,
                   'Log-Likelihood': log_likelihood,
                   'Ljung-Box p-value': ljung_box_p_value,
                   'MAE': mae,
                   'MSE': mse,
                   'RMSE': rmse})

                # Plot residuals
                plt.figure(figsize=(10, 6))
                plt.plot(residuals, label='Residuals')
                plt.title(f'Residuals of SARIMAX Model ({i}, 0, {j}) x ({k}, {l}, 1, 12)')
                plt.xlabel('Time')
                plt.ylabel('Residuals')
                plt.legend()
                plt.show()

                # ACF and PACF of residuals
                plt.figure(figsize=(12, 6))
                plt.subplot(121)
                plot_acf(residuals, lags=60, ax=plt.gca())
                plt.title(f'ACF Plot (Residuals) of SARIMAX Model ({i}, 0, {j}) x ({k}, {l}, 1, 12)')

                plt.subplot(122)
                plot_pacf(residuals, lags=60, ax=plt.gca())
                plt.title(f'PACF Plot (Residuals) of SARIMAX Model ({i}, 0, {j}) x ({k}, {l}, 1, 12)')

                plt.tight_layout()
                plt.show()
                
                count += 1

# Auto ARIMA model
print("\nAuto ARIMA Model")
model_auto_arima = pm.auto_arima(data, seasonal=True, m=12, trace=True, error_action='warn', suppress_warnings=False)
print(model_auto_arima.summary())

# Calculate residuals and metrics
residuals_auto_arima = model_auto_arima.resid()
mae_auto_arima = mean_absolute_error(data, model_auto_arima.predict_in_sample())
mse_auto_arima = mean_squared_error(data, model_auto_arima.predict_in_sample())
rmse_auto_arima = np.sqrt(mse_auto_arima)
ljung_box_result_auto_arima = acorr_ljungbox(residuals_auto_arima, lags=[60], return_df=True)
ljung_box_p_value_auto_arima = ljung_box_result_auto_arima['lb_pvalue'].values[0]

# Append Auto ARIMA results
results_list.append({
   'Model Number': count,
   'Order': model_auto_arima.order,
   'Seasonal Order': model_auto_arima.seasonal_order,
   'AIC': model_auto_arima.aic(),
   'BIC': model_auto_arima.bic(),
   'HQIC': 'N/A',  # Auto ARIMA does not provide HQIC
   'Log-Likelihood': model_auto_arima.arima_res_.llf,
   'Ljung-Box p-value': ljung_box_p_value_auto_arima,
   'MAE': mae_auto_arima,
   'MSE': mse_auto_arima,
   'RMSE': rmse_auto_arima})

# Plot residuals for Auto ARIMA
plt.figure(figsize=(10, 6))
plt.plot(residuals_auto_arima, label='Residuals')
plt.title('Residuals of Auto ARIMA Model')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.legend()
plt.show()

# ACF and PACF of residuals for Auto ARIMA
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(residuals_auto_arima, lags=60, ax=plt.gca())
plt.title('ACF Plot (Residuals) of Auto ARIMA Model')

plt.subplot(122)
plot_pacf(residuals_auto_arima, lags=60, ax=plt.gca())
plt.title('PACF Plot (Residuals) of Auto ARIMA Model')

plt.tight_layout()
plt.show()

count += 1

# Exponential Smoothing
print("\nExponential Smoothing")
model_exp_smoothing = sm.tsa.ExponentialSmoothing(
    data,
    trend='add',
    seasonal='add',
    seasonal_periods=12
).fit()

# Calculate residuals and metrics
fitted_values_exp_smoothing = model_exp_smoothing.fittedvalues
residuals_exp_smoothing = data.values.flatten() - fitted_values_exp_smoothing.values.flatten()
mae_exp_smoothing = mean_absolute_error(data, fitted_values_exp_smoothing)
mse_exp_smoothing = mean_squared_error(data, fitted_values_exp_smoothing)
rmse_exp_smoothing = np.sqrt(mse_exp_smoothing)
ljung_box_result_exp_smoothing = acorr_ljungbox(residuals_exp_smoothing, lags=[60], return_df=True)
ljung_box_p_value_exp_smoothing = ljung_box_result_exp_smoothing['lb_pvalue'].values[0]

# Append Exponential Smoothing results
results_list.append({
   'Model Number': count,
   'Order': 'Exponential Smoothing',
   'Seasonal Order': 'Additive',
   'AIC': model_exp_smoothing.aic,
   'BIC': model_exp_smoothing.bic,
   'HQIC': 'N/A',  # HQIC is not available for Exponential Smoothing
   'Log-Likelihood': 'N/A',  # Log-Likelihood is not available for Exponential Smoothing
   'Ljung-Box p-value': ljung_box_p_value_exp_smoothing,
   'MAE': mae_exp_smoothing,
   'MSE': mse_exp_smoothing,
   'RMSE': rmse_exp_smoothing})

# Plot residuals for Exponential Smoothing
plt.figure(figsize=(10, 6))
plt.plot(residuals_exp_smoothing, label='Residuals')
plt.title('Residuals of Exponential Smoothing Model')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.legend()
plt.show()

# ACF and PACF of residuals for Exponential Smoothing
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(residuals_exp_smoothing, lags=60, ax=plt.gca())
plt.title('ACF Plot (Residuals) of Exponential Smoothing Model')

plt.subplot(122)
plot_pacf(residuals_exp_smoothing, lags=60, ax=plt.gca())
plt.title('PACF Plot (Residuals) of Exponential Smoothing Model')

plt.tight_layout()
plt.show()

# Convert results to DataFrame and save to Excel
results_df = pd.DataFrame(results_list)
results_df.to_excel("model_evaluation_metrics.xlsx", index=False)
'''

model = SARIMAX(data, order=(1, 0, 1), seasonal_order=(3, 0, 1, 12))
results = model.fit()
residuals = results.resid

# Plot residuals
plt.figure(figsize=(10, 6))
plt.plot(residuals, label='Residuals')
plt.title('Residuals of SARIMA Model (1,0,1) (3,0,1,12)')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.legend()
plt.show()

# ACF and PACF of residuals
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(residuals, lags=60, ax=plt.gca())
plt.title('ACF Plot (Residuals) of SARIMA Model (1,0,1) (3,0,1,12) ')

plt.subplot(122)
plot_pacf(residuals, lags=60, ax=plt.gca())
plt.title('PACF Plot (Residuals) of SARIMA Model (1,0,1) (3,0,1,12) ')

plt.tight_layout()
plt.show()