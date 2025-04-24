from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained model and encoders
model_data = None
model_path = 'early_prediction_system_model.pkl'

try:
    with open(model_path, "rb") as file:
        model_data = joblib.load(file)
        print(f"Model loaded successfully from {model_path}.")
except FileNotFoundError:
    print(f"Error: Model file '{model_path}' not found.")
except Exception as e:
    print(f"Error loading model: {str(e)}")

# Extract components from the loaded model data
if model_data:
    diabetes_model = model_data['diabetes_model']
    hypertension_model = model_data['hypertension_model']
    ckd_model = model_data['ckd_model']
    sex_encoder = model_data['sex_encoder']
    activity_encoder = model_data['activity_encoder']
    diabetes_features = model_data['diabetes_features']
    hypertension_features = model_data['hypertension_features']
    ckd_features = model_data['ckd_features']

@app.route('/', methods=['GET'])
def home():
    routes = {
        "Welcome Message": "Welcome to the Luba's Early Prediction System for Chronic Diseases",
        "Available Routes": {
            "/": "Home - Displays this message and available routes",
            "/predict": "POST - Predict disease risks based on input features",
            "/greet?name=YourName": "GET - Greets the user by name"
        }
    }
    return jsonify(routes)


@app.route('/predict', methods=['POST'])
def predict():
    # Validate if the model is loaded
    if model_data is None:
        print("Model is not loaded.")
        return jsonify({'error': 'Model is not loaded. Please check the server configuration.'}), 500

    # Get JSON input data
    input_data = request.json
    if input_data is None:
        print("No input data provided.")
        return jsonify({'error': 'No input data provided. Please send data in JSON format.'}), 400

    # Log the input data for debugging
    print(f"Received input data: {input_data}")

    # Convert input data to a DataFrame
    try:
        patient_df = pd.DataFrame([input_data])
    except Exception as e:
        print(f"Error converting input data to DataFrame: {e}")
        return jsonify({'error': 'Invalid input data format. Ensure all features are provided.'}), 400

    # Encode categorical features
    try:
        if 'sex' in patient_df.columns:
            patient_df['sex'] = sex_encoder.transform(patient_df['sex'])
        if 'physical_activity' in patient_df.columns:
            patient_df['physical_activity'] = activity_encoder.transform(
                patient_df['physical_activity'])
    except Exception as e:
        print(f"Error encoding categorical features: {e}")
        return jsonify({'error': 'Error encoding categorical features. Check input data.'}), 400

    # Perform sequential predictions
    try:
        # Diabetes prediction
        diab_input = patient_df[diabetes_features]
        diabetes_prob = diabetes_model.predict_proba(diab_input)[0][1]
        patient_df['diabetes'] = diabetes_model.predict(diab_input)

        # Hypertension prediction
        hyp_input = patient_df[hypertension_features]
        hypertension_prob = hypertension_model.predict_proba(hyp_input)[0][1]
        patient_df['hypertension'] = hypertension_model.predict(hyp_input)

        # CKD prediction
        ckd_input = patient_df[ckd_features]
        ckd_prob = ckd_model.predict_proba(ckd_input)[0][1]

        # Prepare response
        response = {
            'diabetes_risk': round(diabetes_prob, 2),
            'hypertension_risk': round(hypertension_prob, 2),
            'ckd_risk': round(ckd_prob, 2)
        }
        print(f"Prediction made: {response}")
        return jsonify(response)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction. Check server logs for details.'}), 500


@app.route('/greet', methods=['GET'])
def greet():
    # Get the 'name' query parameter from the request
    name = request.args.get('name')
    
    # Check if name is provided
    if not name:
        return jsonify({'error': 'No name provided. Please include a "name" query parameter.'}), 400
    
    # Return a greeting with the name
    greeting = f"Hello, {name}! Welcome to the Early Prediction System."
    return jsonify({'greeting': greeting})


if __name__ == '__main__':
    port = 10000
    app.run(host='0.0.0.0', port=port)
