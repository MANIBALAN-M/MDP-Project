import pandas as pd
import numpy as np
import requests
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load dataset locally
csv_path = "training_dataset.csv"
try:
    df = pd.read_csv(csv_path)
except Exception as e:
    print(f"Error loading dataset: {e}")
    df = pd.DataFrame()

# Ensure dataset is not empty
if not df.empty:
    # Prepare features and target variables
    X = df[['Heat (°C)', 'Crucible Weight (g)', 'Substance Weight (g)']]
    y = df[['Cycle 1 Time (min)', 'Cycle 2 Time (min)', 'Cycle 3 Time (min)',
            'Cycle 4 Time (min)', 'Without Heating Cycle Time (min)']]

    # Check for NaN or infinite values in features and target
    if np.isnan(X.values).any() or np.isinf(X.values).any():
        print("X contains NaN or Inf values.")
    if np.isnan(y.values).any() or np.isinf(y.values).any():
        print("y contains NaN or Inf values.")

    # Replace NaN or infinite values with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
else:
    print("Dataset is empty. Please check the CSV file.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        heat = float(data.get('heat', 0))
        crucible_weight = float(data.get('crucible_weight', 0))
        substance_weight = float(data.get('substance_weight', 0))

        if df.empty:
            return jsonify({'error': 'Dataset not found!'}), 500

        input_data = scaler.transform([[heat, crucible_weight, substance_weight]])
        prediction = model.predict(input_data)

        result = {
            'Cycle 1 Time': round(prediction[0][0], 2),
            'Cycle 2 Time': round(prediction[0][1], 2),
            'Cycle 3 Time': round(prediction[0][2], 2),
            'Cycle 4 Time': round(prediction[0][3], 2),
            'Without Heating Cycle Time': round(prediction[0][4], 2)
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
