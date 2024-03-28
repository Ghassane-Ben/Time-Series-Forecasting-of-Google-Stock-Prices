import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from pmdarima import auto_arima
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Read the dataset
df = pd.read_csv('GOOG.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Apply ETS Modeling to the  series
ets_model = ExponentialSmoothing(df['Adj Close'], trend='add', seasonal='add', seasonal_periods=12)
ets_fit = ets_model.fit()
ets_forecast= ets_fit.forecast(steps=100)

sarima_model = auto_arima(df['Adj Close'], start_p=0, d=1, start_q=0, max_p=3, max_d=2, max_q=3, seasonal=True,
                         m=12,  # Specify the seasonal periodicity (e.g., 12 for monthly data)
                         D=1,   # Order of seasonal differencing
                         start_P=0, Q=0, max_P=2, max_Q=2,  # Seasonal AR and MA orders
                         trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)

sarima_fit = sarima_model.fit(df['Adj Close'])

sarima_forecast, sarima_confint = sarima_fit.predict(n_periods=100,  return_conf_int=True)

df_stationary = df['Adj Close'].diff().dropna()

# ARIMA Modeling using auto_arima on the series
arima_model = auto_arima(df_stationary, start_p=0, d=1, start_q=0, max_p=3, max_d=2, max_q=3, seasonal=False, 
                         trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
arima_fit = arima_model.fit(df_stationary)

arima_forecast_diff, arima_confint = arima_fit.predict(n_periods=100,  return_conf_int=True)

last_original_value = df['Adj Close'].iloc[-101:-1].values
arima_forecast = last_original_value + np.cumsum(arima_forecast_diff)

# Splitting data into training and test sets for error evaluation
train_size = int(len(df) * 0.8)
test = df.iloc[train_size:]

# Error computation
test_length = len(test)  # Length of the test set
ets_forecast_aligned = ets_fit.forecast(steps=test_length)  # Align ETS forecast length with test set
arima_forecast_aligned = arima_forecast[-100:]  # Last 100 points of ARIMA forecast
sarima_forecast_aligned = sarima_forecast[-100:]  # Last 100 points of ARIMA forecast
test_adj_close_aligned = test['Adj Close'][-100:]  # Last 100 points of test data

# Calculate error metrics
ets_mae = mean_absolute_error(test_adj_close_aligned, ets_forecast_aligned[-100:])
ets_rmse = np.sqrt(mean_squared_error(test_adj_close_aligned, ets_forecast_aligned[-100:]))
arima_mae = mean_absolute_error(test_adj_close_aligned, arima_forecast_aligned)
arima_rmse = np.sqrt(mean_squared_error(test_adj_close_aligned, arima_forecast_aligned))

sarima_mae = mean_absolute_error(test_adj_close_aligned, sarima_forecast_aligned)
sarima_rmse = np.sqrt(mean_squared_error(test_adj_close_aligned, sarima_forecast_aligned))


# Print the error metrics
print(f"ETS Model - MAE: {ets_mae}, RMSE: {ets_rmse}")
print(f"ARIMA Model - MAE: {arima_mae}, RMSE: {arima_rmse}")
print(f"SARIMA Model - MAE: {sarima_mae}, RMSE: {sarima_rmse}")


# Plotting the ETS and ARIMA and SARIMA forecasts along with the original data
plt.figure(figsize=(14, 7))
plt.plot(df['Adj Close'], label='Original Data', color='blue')
plt.plot(pd.date_range(start=df.index[-1], periods=101, freq='B')[1:], arima_forecast_aligned, label='ARIMA Forecast', color='red')
plt.plot(pd.date_range(start=df.index[-1], periods=101, freq='B')[1:], sarima_forecast_aligned, label='SARIMA Forecast', color='black')
plt.plot(pd.date_range(start=df.index[-1], periods=101, freq='B')[1:], ets_forecast_aligned[-100:], label='ETS Forecast', color='green')
plt.title('ETS and ARIMA Forecasts of GOOG Closing Stock Prices')
plt.xlabel('Date')
plt.ylabel('Closing Stock Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


print("ARIMA Forecast Confidence Intervals:")
print(arima_confint)

print("\nSARIMA Forecast Confidence Intervals:")
print(sarima_confint)
