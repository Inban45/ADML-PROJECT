import sys
import os
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add parent directory to path to import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fraud_detection import train_model
from generate_prediction_report import create_prediction_report

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Global variables to hold model and data
model = None
df_context = None

def initialize_model():
    global model, df_context
    print("Initializing Model... This might take a moment.")
    model, df_context = train_model()
    print("Model Initialized.")

# Initialize on startup
initialize_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Convert input to DataFrame (expected by pipeline)
        input_df = pd.DataFrame([data])
        
        # Predict probability
        prob = model.predict_proba(input_df)[0][1]
        
        return jsonify({
            'probability': float(prob),
            'probability_pct': float(prob * 100),
            'is_high_risk': bool(prob > 0.5)
        })
    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/visualize', methods=['POST'])
def visualize():
    try:
        data = request.json
        # Generate report and get base64 string
        image_base64 = create_prediction_report(model, df_context, data, save_path=None)
        
        return jsonify({
            'image': image_base64
        })
    except Exception as e:
        print(f"Error in /visualize: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
