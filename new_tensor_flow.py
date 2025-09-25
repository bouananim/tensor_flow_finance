import yfinance as yf
import pandas as pd
import tensorflow as tf
from tensorflow import keras as kr

# Print versions
print ("please keep an eye on vesrion printed below ")
print("yfinance version:", yf.__version__)
print("pandas version:", pd.__version__)
print("tensorflow version:", tf.__version__)
print("keras version:", kr.__version__)

# 1. Load stock data (e.g., Apple - AAPL)
data = yf.download("AAPL", start="2022-01-01", end="2023-01-01")

print(data.head())  # view first rows

# 2. Preprocess (e.g., use 'Close' price only)
prices = data['Close'].values
prices = prices.reshape(-1, 1)

# Normalize (scaling for neural networks)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
prices_scaled = scaler.fit_transform(prices)

# 3. Convert to TensorFlow dataset
dataset = tf.data.Dataset.from_tensor_slices(prices_scaled)

# Example: create sequences for time-series forecasting
window_size = 10
dataset = dataset.window(window_size, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda w: w.batch(window_size))
dataset = dataset.map(lambda w: (w[:-1], w[-1]))  # (features, label)
dataset = dataset.batch(32).prefetch(1)

for features, label in dataset.take(1):
    print("Features:", features.numpy().flatten())
    print("Label:", label.numpy())
