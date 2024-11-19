# road_accident_severity.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Load the dataset
df = pd.read_csv('road_accidents.csv')

# Step 2: Data Preprocessing
df = pd.get_dummies(df, columns=['Climate', 'RoadQuality', 'DriverState'], drop_first=True)

# Step 3: Define independent and dependent variables
X = df.drop('AccidentImpact', axis=1)  # Independent variables
y = df['AccidentImpact']  # Dependent variable

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train the model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Step 7: Evaluate the model
y_pred = model.predict(X_test_scaled)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Step 8: Save the trained model to a file
joblib.dump(model, 'road_accident_severity_model.pkl')
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler

# Step 9: Example usage of the model for prediction
new_data = pd.DataFrame({
    'DrivingSpeed': [75],
    'Climate_Hot': [1],
    'RoadQuality_Good': [1],
    'DriverState_Sober': [1],
    'DriverState_Tired': [0]
})

# Ensure the new data has all columns that were used during training
# We need to add missing columns (with 0 values) to match the model's expected input
for col in X.columns:
    if col not in new_data.columns:
        new_data[col] = 0

# Reorder columns to match the training data
new_data = new_data[X.columns]

# Step 10: Scale the new data (same scaler used for training data)
new_data_scaled = scaler.transform(new_data)

# Predict accident severity
prediction = model.predict(new_data_scaled)
print(f'Predicted Accident Impact: {prediction[0]}')
