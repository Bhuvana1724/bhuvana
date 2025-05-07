# Blood Pressure Prediction using Linear Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
# You can replace this with your actual dataset
# Example CSV should contain columns like: Age, BMI, Gender (encoded), Systolic, Diastolic
df = pd.read_csv("blood_pressure_data.csv")

# Display basic info
print(df.head())

# Features and target
X = df[['Age', 'BMI', 'Gender']]  # Example features
y = df['Systolic']  # You can also try predicting 'Diastolic'

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

# Plot predictions
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Systolic BP")
plt.ylabel("Predicted Systolic BP")
plt.title("Actual vs Predicted Blood Pressure")
plt.show()
