import tensorflow as tf
import numpy as np

# Print versions
print ("please keep an eye on vesrion printed below ")
--print("yfinance version:", yf.__version__)
--print("pandas version:", pd.__version__)
print("tensorflow version:", tf.__version__)
print("numpy version:", np.__version__)

# Training data: y = 2x - 1
X = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
Y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Build a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Compile model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train model
print("Training...")
model.fit(X, Y, epochs=500, verbose=0)

# Make a prediction
print("Prediction for x=10.0 ->", model.predict([10.0]))
