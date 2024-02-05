import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from datetime import datetime, timedelta
import warnings

def fetch_stock_data(stock_symbol, start_date):
    df = yf.Ticker(stock_symbol).history(start=start_date, period="max")
    df['Date'] = df.index
    df = df[['Date'] + [col for col in df.columns if col != 'Date']]
    df.reset_index(drop=True, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    return df[['Date', 'Close']]

def get_user_stock_symbol():
    stock_symbol = input("Lütfen bir hisse senedi sembolü girin (Örnek: XU100.IS): ")
    return stock_symbol

def split_data(dataframe, test_size):
    position = int(round(len(dataframe) * (1 - test_size)))
    train = dataframe[:position]
    test = dataframe[position:]
    return train, test, position

def scale_data(train_data, test_data):
    scaler_train = MinMaxScaler(feature_range=(0, 1))
    scaler_test = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler_train.fit_transform(train_data)
    test_scaled = scaler_test.fit_transform(test_data)
    return scaler_train, scaler_test, train_scaled, test_scaled

def create_features(data, lookback):
    X, Y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, 0])
        Y.append(data[i, 0])
    return np.array(X), np.array(Y)

def build_model(X_train, y_train, lookback):
    model = Sequential()
    model.add(LSTM(units=50, activation="relu", input_shape=(1, lookback)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model

def train_model(model, X_train, y_train, X_test, y_test, callbacks, epochs, batch_size):
    history = model.fit(
        x=X_train,
        y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        shuffle=False
    )
    return history

def plot_loss(history):
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend(loc="upper right")
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.ylim([0, max(plt.ylim())])
    plt.title("Training and Validation Loss", fontsize=16)
    plt.show()

def evaluate_model(model, X_test, y_test):
    loss = model.evaluate(X_test, y_test, batch_size=20)
    print("\nTest loss:%.1f%%" % (100.0 * loss))

def inverse_transform(scaler, data):
    return scaler.inverse_transform(data)

def make_predictions(model, input_data, scaler):
    predicted_price = model.predict(input_data)
    predicted_price = scaler.inverse_transform(predicted_price)
    return predicted_price[0][0]

def main():
    warnings.filterwarnings("ignore")
    
    start_date = "2010-07-27"
    stock_symbol = get_user_stock_symbol()

    df = fetch_stock_data(stock_symbol, start_date)

    train, test, position = split_data(df, 0.25)

    train_close = train[['Close']].values
    test_close = test[['Close']].values

    scaler_train, scaler_test, train_scaled, test_scaled = scale_data(train_close, test_close)

    lookback = 25
    X_train, y_train = create_features(train_scaled, lookback)
    X_test, y_test = create_features(test_scaled, lookback)

    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    model = build_model(X_train, y_train, lookback)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, verbose=1, mode="min"),
        ModelCheckpoint(filepath="mymodel.h5", monitor="val_loss", mode="min",
                        save_best_only=True, save_weight_only=False, verbose=1)
    ]

    history = train_model(model, X_train, y_train, X_test, y_test, callbacks, epochs=100, batch_size=20)

    plot_loss(history)
    evaluate_model(model, X_test, y_test)

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    train_predict = inverse_transform(scaler_train, train_predict)
    test_predict = inverse_transform(scaler_test, test_predict)

    y_train = inverse_transform(scaler_train, y_train)
    y_test = inverse_transform(scaler_test, y_test)

    train_prediction_df = df[lookback:position]
    train_prediction_df["Predicted"] = train_predict

    test_prediction_df = df[position + lookback:]
    test_prediction_df["Predicted"] = test_predict

    plt.figure(figsize=(14, 5))
    plt.plot(df['Close'], label="Real Values")
    plt.plot(train_prediction_df["Predicted"], color="blue", label="Train Predicted")
    plt.plot(test_prediction_df["Predicted"], color="red", label="Test Predicted")
    plt.xlabel("Time")
    plt.ylabel("Stock Values")
    plt.legend()
    plt.show()

    input_data = df["Close"].values[-lookback:]
    input_data = scaler_test.transform(input_data.reshape(-1, 1))
    input_data = input_data.reshape(1, 1, lookback)

    predicted_price = make_predictions(model, input_data, scaler_test)
    print(f"{datetime.now() } için tahmin edilen kapanış fiyatı:", predicted_price)

if __name__ == "__main__":
    main()
