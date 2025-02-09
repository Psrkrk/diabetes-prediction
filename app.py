from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

# Load trained model
with open('model.pkl', 'rb') as f:
    scaler, model = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        data = [float(request.form[key]) for key in request.form]
        scaled_data = scaler.transform([data])
        prediction = model.predict(scaled_data)[0]
        
        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
