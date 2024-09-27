import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Generate synthetic data
np.random.seed(42)
data_size = 1000

# Simulating system features (CPU usage, RAM usage, Disk activity, etc.)
cpu_usage = np.random.uniform(0, 100, data_size)  # CPU usage percentage
ram_usage = np.random.uniform(0, 100, data_size)  # RAM usage percentage
disk_activity = np.random.uniform(0, 100, data_size)  # Disk activity percentage
battery_usage = np.random.uniform(0, 100, data_size)  # Battery usage percentage

# Simulating energy consumption based on the system features
energy_generated = (
    0.5 * cpu_usage
    + 0.3 * ram_usage
    + 0.1 * disk_activity
    + 0.1 * battery_usage
    + np.random.normal(0, 10, data_size)
)

# Create a DataFrame
data = pd.DataFrame({
    'CPU_Usage': cpu_usage,
    'RAM_Usage': ram_usage,
    'Disk_Activity': disk_activity,
    'Battery_Usage': battery_usage,
    'Energy_Generated': energy_generated
})

# Define feature columns and target column
X = data[['CPU_Usage', 'RAM_Usage', 'Disk_Activity', 'Battery_Usage']]
y = data['Energy_Generated']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Print to verify model coefficients and intercept
print("Model training complete!")
print(f"Model coefficients: {model.coef_}")
print(f"Model intercept: {model.intercept_}")

# Save the trained model to a file
joblib.dump(model, 'energy_model.pkl')

# Print to verify that the model has been saved
print("Model has been saved to 'energy_model.pkl'")
