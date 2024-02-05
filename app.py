from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

app = Flask(__name__)

model = tf.keras.models.load_model('mymodel.h5')
scaler = MinMaxScaler()
lookback = 25

def fetch_stock_data(stock_symbol, start_date):
    df = yf.Ticker(stock_symbol).history(start=start_date, period="max")
    return df[['Close']].values

@app.route('/predict', methods=['GET'])
def predict():
    stock_symbol = request.args.get('symbol')
    stock_data = fetch_stock_data(stock_symbol, start_date="2010-07-27")

    # Gerçek veri ile scaler'ı uyumlu hale getirme
    scaler.fit(stock_data)

    input_data = stock_data[-lookback:]
    predicted_price = make_prediction(input_data)

    response = {
        'symbol': stock_symbol,
        'predictedPrice': predicted_price
    }

    return jsonify(response)

def make_prediction(input_data):
    input_data_scaled = scaler.transform(input_data)
    input_data_scaled = input_data_scaled[-lookback:]  # lookback kadar veriyi al
    input_data_scaled = input_data_scaled.reshape(1, 1, lookback)
    predicted_price = model.predict(input_data_scaled)
    predicted_price = float(scaler.inverse_transform(predicted_price)[0][0])  # float() ile dönüştür
    return predicted_price

if __name__ == '__main__':
    app.run(host="192.168.182.1", port=5000, debug=True)