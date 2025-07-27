# app.py
from flask import Flask, render_template, request
import pickle
import pandas as pd
import sklearn.tree  # Import the tree module from scikit-learn

app = Flask(__name__)

# --- Monkey Patch the DecisionTreeRegressor Class ---
# This ensures that all DecisionTreeRegressor instances have a dummy 'monotonic_cst' attribute.
if not hasattr(sklearn.tree.DecisionTreeRegressor, 'monotonic_cst'):
    setattr(sklearn.tree.DecisionTreeRegressor, 'monotonic_cst', None)

# --- Load the Model and Feature List ---
with open('vehicle_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
feature_cols = model_data['features']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        input_datetime = request.form['datetime']  # Expecting format: YYYY-MM-DD HH:MM:SS
        try:
            dt = pd.to_datetime(input_datetime)
            # Extract features
            hour = dt.hour
            day_of_week = dt.dayofweek
            month = dt.month
            is_weekend = 1 if day_of_week >= 5 else 0

            # Build a data dictionary for model input
            data = {
                'hour': [hour],
                'day_of_week': [day_of_week],
                'month': [month],
                'is_weekend': [is_weekend]
            }
            # Add default values for junction features if they exist
            for col in feature_cols:
                if col.startswith('junction_'):
                    data[col] = [0]

            # Create a DataFrame ensuring the same order of features
            df_input = pd.DataFrame(data)[feature_cols]

            # Make the prediction
            pred = model.predict(df_input)[0]
            prediction = round(pred, 2)
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
