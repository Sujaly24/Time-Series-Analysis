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
data = pd.read_csv("Monthly Rainfall Data.csv", parse_dates=['Month'], index_col='Month')
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

#To find out good models the following loop was run

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

# Convert results to DataFrame and save to Excel
results_df = pd.DataFrame(results_list)
results_df.to_excel("model_evaluation_metrics.xlsx", index=False)
'''

# Define the last 10 years of data
last_10_years = data[-120:]  # Assuming monthly data, 10 years = 120 months

model = SARIMAX(data, order=(1, 0, 1), seasonal_order=(3, 0, 1, 12))
results = model.fit()

# Calculate residuals
residuals = results.resid

# Plot residuals
plt.figure(figsize=(10, 6))
plt.plot(residuals, label='Residuals')
plt.title('Residuals of SARIMA Model (1, 0, 1) x (3, 0, 1, 12)')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.legend()
plt.show()

# ACF and PACF of residuals
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(residuals, lags=60, ax=plt.gca())
plt.title('ACF Plot (Residuals) of SARIMA Model (1, 0, 1) x (3, 0, 1, 12)')

plt.subplot(122)
plot_pacf(residuals, lags=60, ax=plt.gca())
plt.title('PACF Plot (Residuals) of SARIMA Model (1, 0, 1) x (3, 0, 1, 12)')

plt.tight_layout()
plt.show()

# Forecasting for SARIMA Model (1, 0, 1) x (3, 0, 1, 12)
forecast_horizon = 24  # 24 months ahead
sarima_101_301_forecast = results.get_forecast(steps=forecast_horizon)
forecast_ci_101_301 = sarima_101_301_forecast.conf_int()

# Plot
plt.figure(figsize=(12, 8))
plt.plot(last_10_years, label='Observed', color='blue')
plt.plot(sarima_101_301_forecast.predicted_mean, label='Forecast (1, 0, 1) x (3, 0, 1, 12)', color='red')
plt.fill_between(forecast_ci_101_301.index,
                 forecast_ci_101_301.iloc[:, 0],
                 forecast_ci_101_301.iloc[:, 1], color='pink', alpha=0.3)
plt.title('Forecast using SARIMA Model (1, 0, 1) x (3, 0, 1, 12)')
plt.xlabel('Time')
plt.ylabel('Rainfall')
plt.legend()
plt.show()

model = SARIMAX(data, order=(1, 0, 2), seasonal_order=(1, 1, 1, 12))
results = model.fit()

# Calculate residuals
residuals = results.resid

# Plot residuals
plt.figure(figsize=(10, 6))
plt.plot(residuals, label='Residuals')
plt.title('Residuals of SARIMA Model (1, 0, 2) x (1, 1, 1, 12)')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.legend()
plt.show()

# ACF and PACF of residuals
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(residuals, lags=60, ax=plt.gca())
plt.title('ACF Plot (Residuals) of SARIMA Model (1, 0, 2) x (1, 1, 1, 12)')

plt.subplot(122)
plot_pacf(residuals, lags=60, ax=plt.gca())
plt.title('PACF Plot (Residuals) of SARIMA Model (1, 0, 2) x (1, 1, 1, 12)')

plt.tight_layout()
plt.show()

# SARIMA Model (1, 0, 2) x (1, 1, 1, 12)
forecast_horizon = 24
sarima_102_111_forecast = results.get_forecast(steps=forecast_horizon)
forecast_ci_102_111 = sarima_102_111_forecast.conf_int()

plt.figure(figsize=(12, 8))
plt.plot(last_10_years, label='Observed', color='blue')
plt.plot(sarima_102_111_forecast.predicted_mean, label='Forecast (1, 0, 2) x (1, 1, 1, 12)', color='red')
plt.fill_between(forecast_ci_102_111.index,
                 forecast_ci_102_111.iloc[:, 0],
                 forecast_ci_102_111.iloc[:, 1], color='pink', alpha=0.3)
plt.title('Forecast using SARIMA Model (1, 0, 2) x (1, 1, 1, 12)')
plt.xlabel('Time')
plt.ylabel('Rainfall')
plt.legend()
plt.show()

model = SARIMAX(data, order=(3, 0, 2), seasonal_order=(2, 1, 1, 12))
results = model.fit()

# Calculate residuals
residuals = results.resid

# Plot residuals
plt.figure(figsize=(10, 6))
plt.plot(residuals, label='Residuals')
plt.title('Residuals of SARIMA Model (3, 0, 2) x (2, 1, 1, 12)')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.legend()
plt.show()

# ACF and PACF of residuals
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(residuals, lags=60, ax=plt.gca())
plt.title('ACF Plot (Residuals) of SARIMA Model (3, 0, 2) x (2, 1, 1, 12)')

plt.subplot(122)
plot_pacf(residuals, lags=60, ax=plt.gca())
plt.title('PACF Plot (Residuals) of SARIMA Model (3, 0, 2) x (2, 1, 1, 12)')

plt.tight_layout()
plt.show()

# SARIMA Model (3, 0, 2) x (2, 1, 1, 12)
forecast_horizon = 24
sarima_302_211_forecast = results.get_forecast(steps=forecast_horizon)
forecast_ci_302_211 = sarima_302_211_forecast.conf_int()

plt.figure(figsize=(12, 8))
plt.plot(last_10_years, label='Observed', color='blue')
plt.plot(sarima_302_211_forecast.predicted_mean, label='Forecast (3, 0, 2) x (2, 1, 1, 12)', color='red')
plt.fill_between(forecast_ci_302_211.index,
                 forecast_ci_302_211.iloc[:, 0],
                 forecast_ci_302_211.iloc[:, 1], color='pink', alpha=0.3)
plt.title('Forecast using SARIMA Model (3, 0, 2) x (2, 1, 1, 12)')
plt.xlabel('Time')
plt.ylabel('Rainfall')
plt.legend()
plt.show()

model = SARIMAX(data, order=(3, 0, 3), seasonal_order=(1, 0, 1, 12))
results = model.fit()

# Calculate residuals
residuals = results.resid

# Plot residuals
plt.figure(figsize=(10, 6))
plt.plot(residuals, label='Residuals')
plt.title('Residuals of SARIMA Model (3, 0, 3) x (1, 0, 1, 12)')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.legend()
plt.show()

# ACF and PACF of residuals
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(residuals, lags=60, ax=plt.gca())
plt.title('ACF Plot (Residuals) of SARIMA Model (3, 0, 3) x (1, 0, 1, 12)')

plt.subplot(122)
plot_pacf(residuals, lags=60, ax=plt.gca())
plt.title('PACF Plot (Residuals) of SARIMA Model (3, 0, 3) x (1, 0, 1, 12)')

plt.tight_layout()
plt.show()

# SARIMA Model (3, 0, 3) x (1, 0, 1, 12)
forecast_horizon = 24
sarima_303_101_forecast = results.get_forecast(steps=forecast_horizon)
forecast_ci_303_101 = sarima_303_101_forecast.conf_int()

plt.figure(figsize=(12, 8))
plt.plot(last_10_years, label='Observed', color='blue')
plt.plot(sarima_303_101_forecast.predicted_mean, label='Forecast (3, 0, 3) x (1, 0, 1, 12)', color='red')
plt.fill_between(forecast_ci_303_101.index,
                 forecast_ci_303_101.iloc[:, 0],
                 forecast_ci_303_101.iloc[:, 1], color='pink', alpha=0.3)
plt.title('Forecast using SARIMA Model (3, 0, 3) x (1, 0, 1, 12)')
plt.xlabel('Time')
plt.ylabel('Rainfall')
plt.legend()
plt.show()

model = SARIMAX(data, order=(2, 0, 2), seasonal_order=(2, 0, 0, 12))
results = model.fit()

# Calculate residuals
residuals = results.resid

# Plot residuals
plt.figure(figsize=(10, 6))
plt.plot(residuals, label='Residuals')
plt.title('Residuals of SARIMA Model (2, 0, 2) x (2, 0, 0, 12)')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.legend()
plt.show()

# ACF and PACF of residuals
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(residuals, lags=60, ax=plt.gca())
plt.title('ACF Plot (Residuals) of SARIMA Model (2, 0, 2) x (2, 0, 0, 12)')

plt.subplot(122)
plot_pacf(residuals, lags=60, ax=plt.gca())
plt.title('PACF Plot (Residuals) of SARIMA Model (2, 0, 2) x (2, 0, 0, 12)')

plt.tight_layout()
plt.show()

# SARIMA Model (2, 0, 2) x (2, 0, 0, 12)
forecast_horizon = 24
sarima_202_200_forecast = results.get_forecast(steps=forecast_horizon)
forecast_ci_202_200 = sarima_202_200_forecast.conf_int()

plt.figure(figsize=(12, 8))
plt.plot(last_10_years, label='Observed', color='blue')
plt.plot(sarima_202_200_forecast.predicted_mean, label='Forecast (2, 0, 2) x (2, 0, 0, 12)', color='red')
plt.fill_between(forecast_ci_202_200.index,
                 forecast_ci_202_200.iloc[:, 0],
                 forecast_ci_202_200.iloc[:, 1], color='pink', alpha=0.3)
plt.title('Forecast using SARIMA Model (2, 0, 2) x (2, 0, 0, 12)')
plt.xlabel('Time')
plt.ylabel('Rainfall')
plt.legend()
plt.show()

model = sm.tsa.ExponentialSmoothing(
    data,
    trend='add',
    seasonal='add',
    seasonal_periods=12
).fit()

# Calculate residuals
residuals = model.resid

# Plot residuals
plt.figure(figsize=(10, 6))
plt.plot(residuals, label='Residuals')
plt.title('Residuals of Holt-Winters Exponential Smoothing')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.legend()
plt.show()

# ACF and PACF of residuals
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(residuals, lags=60, ax=plt.gca())
plt.title('ACF Plot (Residuals) of Holt-Winters Exponential Smoothing')

plt.subplot(122)
plot_pacf(residuals, lags=60, ax=plt.gca())
plt.title('PACF Plot (Residuals) of Holt-Winters Exponential Smoothing')

plt.tight_layout()
plt.show()

# Holt-Winters Exponential Smoothing
forecast_horizon = 24
hw_forecast = model.forecast(steps=forecast_horizon)
forecast_index = pd.date_range(data.index[-1], periods=forecast_horizon + 1, freq='M')[1:]

plt.figure(figsize=(12, 8))
plt.plot(last_10_years, label='Observed', color='blue')
plt.plot(forecast_index, hw_forecast, label='Holt-Winters Forecast', color='red')
plt.title('Holt-Winters Exponential Smoothing Forecast')
plt.xlabel('Time')
plt.ylabel('Rainfall')
plt.legend()
plt.show()