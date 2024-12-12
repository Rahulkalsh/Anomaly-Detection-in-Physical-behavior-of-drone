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

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'ulg'}

# Ensure necessary directories exist
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/graphs', exist_ok=True)
os.makedirs('static/css', exist_ok=True)

# Load the trained models
model = load_model('models/Regression_50.h5', custom_objects={'mse': 'MeanSquaredError'})
classification_model = load_model('models/Classification_50.h5')  # Load your classification model

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
        
        # Preprocess the uploaded ULog file
        result = preprocess_ulog(filepath)
        if isinstance(result, str):
            return f"Error: {result}"
        
        # Limit the DataFrame to the first 50 rows (for demonstration)
        if len(result) > 5000:
            result = result.iloc[:5000]

        # Apply the model directly without batching
        predictions, real_data = apply_model_without_batches(result, model)
        
        # # Convert to DataFrames for easier handling in the template
        real_data_df = pd.DataFrame(real_data, columns=result.columns)
        predicted_data_df = pd.DataFrame(predictions, columns=result.columns)
        
        # Classify anomalies
        anomalies = classify_anomalies(predicted_data_df)

        # Prepare data for plotting
        graphs = []
        for col in real_data_df.columns:
            trace_real = go.Scatter(
                x=np.arange(len(real_data_df)),
                y=real_data_df[col],
                mode='lines',
                name=f'Real {col}'
            )
            trace_pred = go.Scatter(
                x=np.arange(len(predicted_data_df)),
                y=predicted_data_df[col],
                mode='lines',
                name=f'Predicted {col}'
            )
            trace_anomaly = go.Scatter(
                x=np.where(anomalies == 1)[0],
                y=predicted_data_df[col][anomalies == 1],
                mode='markers',
                marker=dict(color='red'),
                name=f'Anomaly {col}'
            )
            graph = {
                'data': [trace_real, trace_pred, trace_anomaly],
                'layout': {'title': col}
            }
            graphs.append(graph)

        # Serialize graph data to JSON
        graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

        # Render results with graphs
        return render_template('result.html', graphJSON=graphJSON)

    return redirect(url_for('index'))

def apply_model_without_batches(df, model):
    n_past = 200
    n_future = 50
    
    # Scale the data
    scaler = RobustScaler().fit(df)
    data_scaled = scaler.transform(df)

    # Create sequences
    X, y = create_sequences(data_scaled, n_past, n_future)
    print(X.shape)
    print(y.shape)
    # Apply the model directly without batching
    y_pred = model.predict(X)

    # Reshape and inverse transform the predictions
    y_test_reshaped = y.reshape(-1, y.shape[2])
    y_pred_reshaped = y_pred.reshape(-1, y_pred.shape[2])

    y_test_original = scaler.inverse_transform(y_test_reshaped)
    y_pred_original = scaler.inverse_transform(y_pred_reshaped)

    return y_pred_original, y_test_original

def create_sequences(data, n_past, n_future):
    X, y = [], []
    num_samples = len(data) - n_past - n_future + 1
    i = 0
    while(i < num_samples):
        X.append(data[i: i + n_past, :])
        y.append(data[i + n_past: i + n_past + n_future, :])
        i = i + 10
    # for i in range(num_samples):
    #     X.append(data[i: i + n_past, :])
    #     y.append(data[i + n_past: i + n_past + n_future, :])
    return np.array(X), np.array(y)

def classify_anomalies(predicted_data_df):
    # Preprocess the predicted data
    scale_columns = predicted_data_df.columns
    scaler = RobustScaler().fit(predicted_data_df[scale_columns])
    predicted_data_df.loc[:, scale_columns] = scaler.transform(predicted_data_df[scale_columns])
    
    # Create dataset for testing
    TIME_STEPS = 50
    STEP = 20
    X_test = create_dataset(predicted_data_df[scale_columns], TIME_STEPS, STEP)
    
    # Ensure X_test is 3-dimensional
    if X_test.ndim != 3:
        raise ValueError(f"Expected X_test to be 3-dimensional, but got {X_test.ndim} dimensions")

    # Prepare input for testing
    decoder_input_data_test = np.zeros_like(X_test)
    if decoder_input_data_test.shape[1] < 2:
        raise ValueError("Not enough time steps for shifting decoder input data")
    decoder_input_data_test[:, 1:, :] = X_test[:, :-1, :] 

    # Predict and classify anomalies
    threshold = 0.3  # Example threshold
    y_pred_prob = classification_model.predict([X_test, decoder_input_data_test])
    y_pred = (y_pred_prob > threshold).astype(int).flatten()
    
    # Reshape y_pred to match the length of predicted_data_df
    anomaly_indices = np.arange(len(y_pred)) * STEP
    anomalies = np.zeros(len(predicted_data_df), dtype=int)
    anomalies[anomaly_indices] = y_pred[:len(anomaly_indices)]

    return anomalies

def create_dataset(data, time_steps=1, step=1):
    Xs = []
    num_samples = len(data) - time_steps + 1
    for i in range(0, num_samples, step):
        # Ensure we do not exceed the bounds of the data
        if i + time_steps <= len(data):
            Xs.append(data[i: i + time_steps])
    Xs_array = np.array(Xs)
    
    # Debug: print the shape of the resulting array
    print(f"Shape of dataset created: {Xs_array.shape}")
    
    return Xs_array





# from flask import Flask, request, render_template, redirect, url_for
# import os
# import pandas as pd
# from werkzeug.utils import secure_filename
# from preprocessing import preprocess_ulog
# from tensorflow.keras.models import load_model
# import numpy as np
# from sklearn.preprocessing import RobustScaler
# import plotly.graph_objs as go
# import plotly
# import json

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads'
# app.config['ALLOWED_EXTENSIONS'] = {'ulg'}

# # Ensure necessary directories exist
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# os.makedirs('static/graphs', exist_ok=True)
# os.makedirs('static/css', exist_ok=True)

# # Load the trained model
# model = load_model('models/trained_model_final.h5', custom_objects={'mse': 'MeanSquaredError'})

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return redirect(request.url)
#     file = request.files['file']
#     if file.filename == '':
#         return redirect(request.url)
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
        
#         # Preprocess the uploaded ULog file
#         result = preprocess_ulog(filepath)
#         if isinstance(result, str):
#             return f"Error: {result}"
        
#         # Limit the DataFrame to the first 200 rows
#         if len(result) > 1000:
#             result = result.iloc[:1000]

#         # Apply the model directly without batching
#         predictions, real_data = apply_model_without_batches(result, model)
        
#         # Convert to DataFrames for easier handling in the template
#         real_data_df = pd.DataFrame(real_data, columns=result.columns)
#         predicted_data_df = pd.DataFrame(predictions, columns=result.columns)
        
#         # Prepare data for plotting
#         graphs = []
#         for col in real_data_df.columns:
#             trace_real = go.Scatter(
#                 x=np.arange(len(real_data_df)),
#                 y=real_data_df[col],
#                 mode='lines',
#                 name=f'Real {col}'
#             )
#             trace_pred = go.Scatter(
#                 x=np.arange(len(predicted_data_df)),
#                 y=predicted_data_df[col],
#                 mode='lines',
#                 name=f'Predicted {col}'
#             )
#             graph = {
#                 'data': [trace_real, trace_pred],
#                 'layout': {'title': col}
#             }
#             graphs.append(graph)

#         # Serialize graph data to JSON
#         graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

#         # Render results with graphs
#         return render_template('result.html', graphJSON=graphJSON)

#     return redirect(url_for('index'))

# def apply_model_without_batches(df, model):
#     n_past = 20
#     n_future = 5
    
#     # Scale the data
#     scaler = RobustScaler().fit(df)
#     data_scaled = scaler.transform(df)

#     # Create sequences
#     X, y = create_sequences(data_scaled, n_past, n_future)
    
#     # Apply the model directly without batching
#     y_pred = model.predict(X)

#     # Reshape and inverse transform the predictions
#     y_test_reshaped = y.reshape(-1, y.shape[2])
#     y_pred_reshaped = y_pred.reshape(-1, y_pred.shape[2])

#     y_test_original = scaler.inverse_transform(y_test_reshaped)
#     y_pred_original = scaler.inverse_transform(y_pred_reshaped)

#     return y_pred_original, y_test_original

# def create_sequences(data, n_past, n_future):
#     X, y = [], []
#     num_samples = len(data) - n_past - n_future + 1
#     for i in range(num_samples):
#         X.append(data[i: i + n_past, :])
#         y.append(data[i + n_past: i + n_past + n_future, :])
#         i = i + n_future
#     return np.array(X), np.array(y)

# if __name__ == '__main__':
#     app.run(debug=True)
