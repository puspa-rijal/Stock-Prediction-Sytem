import os
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, url_for
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import pickle
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/predict_stock', methods=['GET'])
def predict_stock():
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({'error': 'Ticker is required'}), 400

    model_path = f'models/{ticker}_model.h5'
    scaler_path = f'scalers/{ticker}_scaler.pkl'
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return jsonify({'error': 'Model or scaler not found'}), 404

    model = load_model(model_path)

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    df = yf.download(ticker, period='10y')
    data = df['Close'].values.reshape(-1, 1)
    scaled_data = scaler.transform(data)

    time_step = 60
    X_test = [scaled_data[-time_step:]]
    X_test = np.array(X_test).reshape((1, time_step, 1))

    prediction = model.predict(X_test)
    prediction = scaler.inverse_transform(prediction)
    
    # Plotting the prediction
    future_steps = 10
    future_predictions = []
    current_data = scaled_data[-time_step:]
    for _ in range(future_steps):
        next_pred = model.predict(current_data.reshape((1, time_step, 1)))
        future_predictions.append(next_pred[0, 0])
        current_data = np.append(current_data[1:], next_pred[0, 0]).reshape(time_step, 1)

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Close'], label='Actual Price')
    plt.plot([df.index[-1] + pd.Timedelta(days=i) for i in range(1, future_steps + 1)], future_predictions, label='Predicted Price', color='red')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'{ticker} Stock Price Prediction')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    # Get the actual stock price now
    actual_price_now = df['Close'].iloc[-1]
    
    # Get the predicted prices for the next three days
    next_three_days_predictions = future_predictions[:3].flatten().tolist()

    return jsonify({
        'actual_price_now': actual_price_now,
        'next_three_days_predictions': next_three_days_predictions,
        'plot_url': f'data:image/png;base64,{plot_url}'
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)
