import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

# load trained pickle
with open(file="model.pkl", mode="rb") as f:
    model = pickle.load(f)

# create flask app
app = Flask(__name__)

@app.route("/")
def Home():
    return render_template("index.html")

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        input_data = {
            'Age': int(request.form['Age']),
            'Gender': request.form['Gender'],
            'Smoking': request.form['Smoking'],
            'Hx_Radiothreapy': request.form['Hx_Radiothreapy'],
            'Thyroid_Function': request.form['Thyroid_Function'],
            'Physical_Examination': request.form['Physical_Examination'],
            'Adenopathy': request.form['Adenopathy'],
            'Pathology': request.form['Pathology'],
            'Focality': request.form['Focality'],
            'Risk': request.form['Risk'],
            'Stage': request.form['Stage'],
            'Response': request.form['Response']
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Make prediction
        prediction = model.predict(input_df)[0]
        output = "Yes, it will recurred" if prediction == 1 else "No, it won't recurred"

        return render_template('index.html', prediction=output)
    
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

# Run the app
if __name__ == '__main__':
    app.run(debug=True)