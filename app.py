import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request

# Load the saved model
loaded_model = joblib.load('RFG_Regression_model.joblib')

# Access the components of the loaded model
model = loaded_model['model']
imputer = loaded_model['imputer']
scaler = loaded_model['scaler']
encoder = loaded_model['encoder']
numeric_cols = loaded_model['numeric_cols']
categorical_cols = loaded_model['categorical_cols']
encoded_cols = loaded_model['encoded_cols']

# Define a function to predict a single input
def predict_input(model, single_input):
    input_df = pd.DataFrame([single_input])
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    input_df[encoded_cols] = encoder.transform(input_df[categorical_cols])
    X_input = input_df[numeric_cols + encoded_cols]
    pred = model.predict(X_input)[0]
    return pred

# Default feature values (replace with actual default values from your data if available)
default_input = {
    'type': 'apartment',
    'sqfeet': 1000,
    'beds': 2,
    'baths': 1,
    'cats_allowed': 1,
    'dogs_allowed': 0,
    'smoking_allowed': 0,
    'wheelchair_access': 0,
    'electric_vehicle_charge': 0,
    'comes_furnished': 0,
    'laundry_options': 'w/d in unit',
    'parking_options': 'off-street parking',
    'lat': 34.0522,
    'long': -118.2437,
    'state': 'ca'
}

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    # Initialize `submitted_values` to default input values
    submitted_values = default_input.copy()
    predicted_price = None

    if request.method == 'POST':
        try:
            # Capture submitted form data
            input_data = request.form.to_dict()

            # Convert string inputs to appropriate types
            for key in input_data:
                if key in ['sqfeet', 'beds', 'baths', 'cats_allowed', 'dogs_allowed', 'smoking_allowed', 'wheelchair_access', 'electric_vehicle_charge', 'comes_furnished']:
                    input_data[key] = int(input_data[key])
                if key in ['lat', 'long']:
                    input_data[key] = float(input_data[key])

            # Update submitted_values with the user input
            submitted_values.update(input_data)

            # Predict price using the updated input
            predicted_price = predict_input(model, submitted_values)
        except ValueError:
            predicted_price = "Error: Invalid input. Please enter numeric values for sqfeet, beds, baths, etc."
        except Exception as e:
            predicted_price = f"Error: {e}"

    # Pass submitted values to the template along with the prediction
    return render_template('index.html', submitted_values=submitted_values, prediction=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
