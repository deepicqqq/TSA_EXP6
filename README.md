# Ex.No: 6   HOLT WINTERS METHOD

### AIM:
To create and implement Holt Winter's Method Model using python for
DailyDelhiClimateTest.csv

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
import pandas as pd\
import matplotlib.pyplot as plt\
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# Load the dataset
data = pd.read_csv('DailyDelhiClimateTest.csv', parse_dates=['date'], index_col='date')

# Inspect the first few rows of the dataset
# Fit the Holt-Winters model
hw_model = ExponentialSmoothing(
    data['meantemp'],           # The time series data (mean temperature)
    trend='additive',           # Additive trend (or 'multiplicative' if needed)
    seasonal='additive',        # Additive seasonality (can be 'multiplicative' if appropriate)
    seasonal_periods=365        # Assuming daily data with yearly seasonality (365 days)
).fit()

# Add the fitted values (in-sample predictions) to the dataset
data['HW_Fitted'] = hw_model.fittedvalues

# Plot the observed data and the fitted values
plt.figure(figsize=(10, 6))\
plt.plot(data['meantemp'], label='Observed')\
plt.plot(data['HW_Fitted'], label='Holt-Winters Fitted', color='red')\
plt.title('Holt-Winters Model Fitting for Mean Temperature in Delhi')\
plt.xlabel('Date')\
plt.ylabel('Temperature (°C)')\
plt.legend()\
plt.show()\
print(data.head())
# Forecast future values (e.g., next 30 days)
forecast = hw_model.forecast(steps=30)

# Plot the observed data along with the forecast
plt.figure(figsize=(10, 6))\
plt.plot(data['meantemp'], label='Observed')\
plt.plot(forecast, label='Forecast', color='green')\
plt.title('Holt-Winters Forecast for Mean Temperature in Delhi')\
plt.xlabel('Date')\
plt.ylabel('Temperature (°C)')\
plt.legend()\
plt.show()

# Print the forecasted values
print(forecast)
from sklearn.metrics import mean_squared_error

# Calculate the Mean Squared Error between actual data and fitted values
mse = mean_squared_error(data['meantemp'], data['HW_Fitted'])\
print(f'Mean Squared Error: {mse:.2f}')# Plot the temperature data\
plt.figure(figsize=(10, 6))\
plt.plot(data['meantemp'], label='Mean Temperature (°C)')\
plt.title('Daily Mean Temperature in Delhi')\
plt.xlabel('Date')\
plt.ylabel('Temperature (°C)')\
plt.legend()\
plt.show()

### OUTPUT:

TEST_PREDICTION
![Screenshot 2024-10-16 100052](https://github.com/user-attachments/assets/8979e509-bb84-4466-92d2-3910ff931702)

FINAL_PREDICTION

![Screenshot 2024-10-16 100035](https://github.com/user-attachments/assets/f8c72bb2-ac8e-4514-b6b4-35b7fd04d295)

### RESULT:
Thus the program run successfully based on the Holt Winters Method model for DailyDelhiClimateTest.csv .
