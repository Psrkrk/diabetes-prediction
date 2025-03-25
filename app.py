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
        # Extract JSON data from request
        data = request.get_json()
        
        # Convert all values to float
        input_data = np.array([[float(data[key]) for key in data]])
        
        # Scale input data
        scaled_data = scaler.transform(input_data)
        
        # Predict outcome
        prediction = model.predict(scaled_data)[0]
        
        return jsonify({'prediction': int(prediction)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)