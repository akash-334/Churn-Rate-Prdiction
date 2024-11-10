# Importing necessary libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')  # Make sure to have an 'index.html' in the 'templates' folder

# Define the predict route for form submission
@app.route('/predict', methods=['POST'])
def predict():
    # For rendering results on HTML GUI
    int_features = [float(x) for x in request.form.values()]  # Collect input values
    final_features = np.array(int_features).reshape(1, -1)  # Reshape for prediction

    # Scale the input features
    scaled_features = scaler.transform(final_features)

    # Make prediction
    prediction = model.predict(scaled_features)

    # Prepare output based on prediction
    output = 'Yes' if prediction[0] == 1 else 'No'

    # Return the result to index.html
    return render_template('index.html', prediction_text='Customer Churn Prediction: {}'.format(output))

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
