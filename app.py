from flask import Flask, request, render_template, redirect, url_for
import os
import pandas as pd
from werkzeug.utils import secure_filename
from preprocessing import preprocess_ulog
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import RobustScaler
import plotly.graph_objs as go
import plotly
import json
import logging

tf.get_logger().setLevel(logging.ERROR)
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'ulg'}

# Ensure necessary directories exist
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/graphs', exist_ok=True)
os.makedirs('static/css', exist_ok=True)

# Load models
model = load_model('models/Regression_50.h5', custom_objects={'mse': 'MeanSquaredError'})
classification_model = load_model('models/Classification_50.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = preprocess_ulog(filepath)
        if isinstance(result, str):
            return f"Error: {result}"
        
        if len(result) > 5000:
            result = result.iloc[:5000]

        predictions, real_data = apply_model_without_batches(result, model)
        real_data_df = pd.DataFrame(real_data, columns=result.columns)
        predicted_data_df = pd.DataFrame(predictions, columns=result.columns)
        
        anomalies = classify_anomalies(predicted_data_df)

        graphs = []
        for col in real_data_df.columns:
            graphs.append({
                'data': [
                    go.Scatter(x=np.arange(len(real_data_df)), y=real_data_df[col], mode='lines', name=f'Real {col}'),
                    go.Scatter(x=np.arange(len(predicted_data_df)), y=predicted_data_df[col], mode='lines', name=f'Predicted {col}'),
                    go.Scatter(x=np.where(anomalies == 1)[0], y=predicted_data_df[col][anomalies == 1], mode='markers', marker=dict(color='red'), name=f'Anomaly {col}')
                ],
                'layout': {'title': col}
            })

        graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
        return render_template('result.html', graphJSON=graphJSON)

    return redirect(url_for('index'))

def apply_model_without_batches(df, model):
    n_past = 200
    n_future = 50
    scaler = RobustScaler().fit(df)
    data_scaled = scaler.transform(df)

    X, y = create_sequences(data_scaled, n_past, n_future)
    y_pred = model.predict(X)

    y_test_original = scaler.inverse_transform(y.reshape(-1, y.shape[2]))
    y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, y_pred.shape[2]))

    return y_pred_original, y_test_original

def create_sequences(data, n_past, n_future):
    X, y = [], []
    num_samples = len(data) - n_past - n_future + 1
    for i in range(0, num_samples, 10):  # Adjust step size if needed
        X.append(data[i: i + n_past, :])
        y.append(data[i + n_past: i + n_past + n_future, :])
    return np.array(X), np.array(y)

def classify_anomalies(predicted_data_df):
    scaler = RobustScaler().fit(predicted_data_df)
    data_scaled = scaler.transform(predicted_data_df)
    X_test = create_dataset(data_scaled, time_steps=50, step=20)

    decoder_input_data_test = np.zeros_like(X_test)
    decoder_input_data_test[:, 1:, :] = X_test[:, :-1, :]

    y_pred_prob = classification_model.predict([X_test, decoder_input_data_test])
    y_pred = (y_pred_prob > 0.3).astype(int).flatten()

    anomalies = np.zeros(len(predicted_data_df), dtype=int)
    anomaly_indices = np.arange(len(y_pred)) * 20
    anomalies[anomaly_indices] = y_pred[:len(anomaly_indices)]

    return anomalies

def create_dataset(data, time_steps=1, step=1):
    Xs = [data[i: i + time_steps] for i in range(0, len(data) - time_steps + 1, step)]
    return np.array(Xs)
