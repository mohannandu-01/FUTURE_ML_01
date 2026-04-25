# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load Dataset
df = pd.read_csv("sales_data.csv")

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sort by date
df = df.sort_values('Date')

# Create time index feature
df['Day_Number'] = np.arange(len(df))

# Features and Target
X = df[['Day_Number']]
y = df['Sales']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Model Evaluation
mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", mae)

# Forecast Future Sales
future_days = 10
future_index = np.arange(len(df), len(df) + future_days).reshape(-1, 1)

future_predictions = model.predict(future_index)

# Create future dates
last_date = df['Date'].iloc[-1]
future_dates = pd.date_range(last_date, periods=future_days + 1)[1:]

# Plot Actual vs Predicted
plt.figure()

plt.plot(df['Date'], df['Sales'], label="Actual Sales")

plt.plot(df['Date'].iloc[-len(predictions):],
         predictions,
         label="Predicted Sales")

plt.plot(future_dates,
         future_predictions,
         label="Future Forecast")

plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Sales Forecasting")

plt.legend()
plt.show()

# Print Future Forecast
forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Sales": future_predictions
})

print("\nFuture Sales Forecast:")
print(forecast_df)
