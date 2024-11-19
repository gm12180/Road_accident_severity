# Road Accident Severity Prediction
Student ID; BSE-01-0062/2024
This repository contains the implementation of a machine learning model to predict the severity of road accidents. The model is built using **Linear Regression** and utilizes a variety of features such as weather conditions, traffic volume, road type, and time of day to predict accident severity. The goal of this project is to demonstrate how data-driven models can be used to enhance road safety, particularly in underdeveloped countries.

## Assignment Overview

The purpose of this assignment is to create a predictive model for road accident severity based on various factors. The task involves:
1. Analyzing relevant datasets and identifying features that influence accident severity.
2. Building a **Linear Regression** model to predict the accident severity.
3. Saving the trained model for future use.
4. Providing an example to demonstrate how the model can predict accident severity using hypothetical data.
5. Explaining how this model could help improve traffic safety and accident prevention, particularly in underdeveloped countries.

## Dataset

The dataset used for this assignment includes various attributes that contribute to the severity of road accidents. These attributes are categorized as:

- **Weather Conditions**: Represents the weather on the day of the accident (e.g., Sunny, Rainy, Foggy).
- **Traffic Volume**: The level of traffic at the time of the accident (e.g., Light, Moderate, Heavy).
- **Road Conditions**: The state of the road (e.g., Wet, Dry, Icy).
- **Time of Day**: When the accident occurred (e.g., Morning, Afternoon, Night).

These factors are used as **independent variables** to predict the **dependent variable**, which is the severity of the accident (e.g., Minor, Moderate, Severe).

## Installation

To run the project, you will need Python installed along with the required libraries. You can set up the project environment by running:

1. Clone this repository:

    ```bash
    git clone https://github.com/yourusername/Road_Accident_Severity.git
    cd Road_Accident_Severity
    ```

2. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- joblib

## Usage

After installing the dependencies, you can use the provided Python scripts to run the model.

1. **Preprocess the Data**: The first step is to load and preprocess the data, ensuring that there are no missing values and that categorical variables are encoded correctly.
2. **Train the Model**: The next step is to train the **Linear Regression** model using the preprocessed dataset.
3. **Evaluate the Model**: Evaluate the performance of the model using evaluation metrics such as **Mean Squared Error (MSE)** and **R-squared**.
4. **Save the Model**: Once trained, the model is saved to a file using `joblib`, so it can be reused for future predictions.

### Example Code to Predict Accident Severity:

```python
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model = joblib.load('road_accident_severity_model.pkl')
scaler = joblib.load('scaler.pkl')

# Hypothetical input for prediction (e.g., rainy weather, moderate traffic)
new_data = [[1, 2, 1, 0]]  # Example: [Rainy, Moderate Traffic, Wet Road, Morning]

# Scale the data and make a prediction
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)

print(f"Predicted Accident Severity: {prediction[0]}")
