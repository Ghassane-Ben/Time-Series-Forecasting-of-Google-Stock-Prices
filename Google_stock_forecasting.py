import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from pmdarima import auto_arima
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Read the dataset
df = pd.read_csv('GOOG.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Extract 'Year' and 'Month' from the 'Date' column
df['Year'] = df.index.year
df['Month'] = df.index.month

# Decompose the time series
result = seasonal_decompose(df['Adj Close'], model='additive', period=252)
result.plot()
plt.show()

# Monthly Growth Plot
unique_years = df['Year'].unique()
plt.figure(figsize=(14, 7))
for month in range(1, 13):
    monthly_data = df[df['Month'] == month]['Adj Close'].values
    plt.scatter([month] * len(monthly_data), monthly_data, marker='o', alpha=0.6, label=f'Month {month}')
means = df.groupby('Month')['Adj Close'].mean()
plt.plot(range(1, 13), means, color='black', linestyle='--', linewidth=2, label='Monthly Mean')
plt.title('Monthly Growth of Adjusted Close Prices Over Years')
plt.xlabel('Month')
plt.ylabel('Adjusted Close Price')
plt.xticks(ticks=range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend(title='Legend', loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# ADF Test and ACF Plot
result = adfuller(df['Adj Close'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:', result[4])
plot_acf(df['Adj Close'], lags=40)
plt.title('Autocorrelation Function')
plt.show()

# ARIMA Modeling using auto_arima
arima_model = auto_arima(df['Adj Close'], seasonal=False, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
arima_fit = arima_model.fit(df['Adj Close'])

# ETS Modeling without seasonality
ets_model = ExponentialSmoothing(df['Adj Close'], trend='add')
ets_fit = ets_model.fit()

# Splitting data into training and test sets
forecast_steps = 100
train = df['Adj Close'][:-forecast_steps]
test = df['Adj Close'][-forecast_steps:]

# Forecasting
arima_forecast = arima_fit.predict(n_periods=forecast_steps)
ets_forecast = ets_fit.forecast(steps=forecast_steps)

# Calculate performance metrics
arima_mae = mean_absolute_error(test, arima_forecast)
arima_rmse = np.sqrt(mean_squared_error(test, arima_forecast))

ets_mae = mean_absolute_error(test, ets_forecast)
ets_rmse = np.sqrt(mean_squared_error(test, ets_forecast))

print(f"ARIMA Model - MAE: {arima_mae}, RMSE: {arima_rmse}")
print(f"ETS Model - MAE: {ets_mae}, RMSE: {ets_rmse}")

# Plotting
forecast_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_steps)
context_points = 50
plt.figure(figsize=(14, 7))
plt.plot(df['Adj Close'][-context_points:], label='Original Data', color='blue', linestyle='dotted')
plt.plot(forecast_dates, arima_forecast, label='ARIMA Forecast', color='red')
plt.plot(forecast_dates, ets_forecast, label='ETS Forecast', color='purple')
plt.title('Forecasts of GOOG Closing Stock Prices')
plt.xlabel('Date')
plt.ylabel('Closing Stock Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
