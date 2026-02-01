from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import os

app = Flask(__name__)

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, '..', 'models')

# Load the trained Ridge model and StandardScaler
try:
    ridge_model_path = os.path.join(models_dir, 'ridge.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    
    print(f"📂 Loading model from: {ridge_model_path}")
    print(f"📂 Loading scaler from: {scaler_path}")
    
    ridge_model = pickle.load(open(ridge_model_path, "rb"))
    standard_scaler = pickle.load(open(scaler_path, "rb"))
    
    print(f"✅ Model loaded successfully!")
    print(f"✅ Scaler loaded successfully! Expecting {standard_scaler.n_features_in_} features")
    
except Exception as e:
    print(f"❌ Error loading model/scaler: {e}")
    ridge_model = None
    standard_scaler = None

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests from the form"""
    if ridge_model is None or standard_scaler is None:
        return render_template('index.html', 
                             prediction_text="Error: Model not loaded properly",
                             prediction_color="danger")
    
    try:
        # Extract 9 features from the form (FWI is OUTPUT, not input!)
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        DC = float(request.form.get('DC'))
        ISI = float(request.form.get('ISI'))
        BUI = float(request.form.get('BUI'))
        # Note: FWI is NOT here - it's what we're predicting!
        
        # Create feature array with 9 features
        features = np.array([[Temperature, RH, Ws, Rain, 
                             FFMC, DMC, DC, ISI, BUI]])
        
        # Scale the features using the pre-fitted scaler
        scaled_features = standard_scaler.transform(features)
        
        # Make prediction (this is the FWI value or fire probability)
        prediction = ridge_model.predict(scaled_features)[0]
        
        # Format the prediction result
        if hasattr(ridge_model, 'predict_proba'):
            # Classification model (0=no fire, 1=fire)
            if prediction > 0.5:
                result = f"🔥 FIRE DETECTED (Probability: {prediction:.2%})"
                color = "danger"
            else:
                result = f"✅ NO FIRE (Probability: {(1-prediction):.2%})"
                color = "success"
        else:
            # Regression model (predicting FWI value)
            if prediction < 5:
                risk = "Low"
                color = "success"
            elif prediction < 10:
                risk = "Moderate"
                color = "warning"
            elif prediction < 20:
                risk = "High"
                color = "danger"
            else:
                risk = "Extreme"
                color = "dark"
            
            result = f"Predicted FWI: {prediction:.2f} → {risk} Fire Danger"
        
        return render_template('index.html', 
                             prediction_text=result,
                             prediction_color=color,
                             show_result=True)
        
    except ValueError as e:
        return render_template('index.html', 
                             prediction_text=f"Error: Please enter valid numbers",
                             prediction_color="warning",
                             show_result=True)
    except Exception as e:
        return render_template('index.html', 
                             prediction_text=f"Error: {str(e)}",
                             prediction_color="danger",
                             show_result=True)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic access"""
    if ridge_model is None or standard_scaler is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        # Extract 9 features
        features = np.array([[
            float(data['Temperature']),
            float(data['RH']),
            float(data['Ws']),
            float(data['Rain']),
            float(data['FFMC']),
            float(data['DMC']),
            float(data['DC']),
            float(data['ISI']),
            float(data['BUI'])
        ]])
        
        # Scale and predict
        scaled_features = standard_scaler.transform(features)
        prediction = float(ridge_model.predict(scaled_features)[0])
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'message': 'Prediction successful'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)