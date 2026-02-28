from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

app = Flask(__name__)

# Get the absolute path to the models folder within the project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Root directory of your Flask app
MODELS_DIR = os.path.join(BASE_DIR, 'models')  # Path to the "models" folder

# Load the .h5 model file
MODEL_PATH = os.path.join(MODELS_DIR, 'bone_tumor_model.h5')
model = load_model(MODEL_PATH)

# Load the CSV file
CSV_PATH = os.path.join(MODELS_DIR, 'Bonetumor.csv')
df = pd.read_csv(CSV_PATH)

# Continue with the rest of your preprocessing and prediction logic...
categorical_cols = ['Sex', 'Grade', 'Histological type', 'MSKCC type', 'Site of primary STS', 'Treatment']
category_values = {col: sorted(df[col].dropna().unique().tolist()) for col in categorical_cols}

# Fit scaler to 'Age' column
scaler = StandardScaler()
scaler.fit(df[["Age"]])

# Preprocess input to match model
def preprocess_input(form_data):
    age = float(form_data['Age'])
    age_scaled = scaler.transform(np.array(age).reshape(-1, 1))[0][0]

    all_columns = ['Age'] + list(np.concatenate([
        [f"{col}_{val}" for val in category_values[col]] for col in categorical_cols
    ]))

    row = dict.fromkeys(all_columns, 0)
    row['Age'] = age_scaled

    for col in categorical_cols:
        key = f"{col}_{form_data[col]}"
        if key in row:
            row[key] = 1

    return np.array([list(row.values())])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        form_data = request.form
        input_data = preprocess_input(form_data)
        prediction = model.predict(input_data)
        predicted_status = ['AWD', 'D', 'NED'][np.argmax(prediction)]
        return render_template('result.html', status=predicted_status)
    
    return render_template('index.html', categories=category_values)

if __name__ == '__main__':
    app.run(debug=True)
