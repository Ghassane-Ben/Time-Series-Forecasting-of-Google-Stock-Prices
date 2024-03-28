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

# Make the series stationary
df_stationary = df['Adj Close'].diff().dropna()

# ADF Test on the differenced series
adf_result_stationary = adfuller(df_stationary)
print('ADF Statistic (Differenced):', adf_result_stationary[0])
print('p-value (Differenced):', adf_result_stationary[1])
print('Critical Values (Differenced):', adf_result_stationary[4])

# ACF Plot on the differenced series
plot_acf(df_stationary, lags=40)
plt.title('Autocorrelation Function on Differenced Series')
plt.show()
