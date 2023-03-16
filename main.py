import numpy as np
from keras import Sequential
from keras.layers import Dense
from scipy import stats

def black_scholes_call_price(S, K, r, q, T, sigma):
    d1 = (np.log(S / K) + (r - q + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * np.exp(-q * T) * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    return call_price

def black_scholes_nn(S, K, r, q, T, sigma):
    # Create input data
    x_train = np.array([[S, K, r, q, T]])

    # Define neural network architecture
    model = Sequential()
    model.add(Dense(units=10, activation='relu', input_shape=(5,)))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=1, activation=None))

    # Compile model
    model.compile(loss='mse', optimizer='adam')

    # Train model
    model.fit(x_train, np.array([black_scholes_call_price(S, K, r, q, T, sigma)]),
              epochs=200, batch_size=256, verbose=0)
    return model

def predict_price(model, S, K, r, q, T, sigma):
    input_data = np.array(
        [[np.log(S / K) / (sigma * np.sqrt(T)) + (r - q + sigma ** 2 / 2) * T / (sigma * np.sqrt(T))]])
    input_data = input_data.reshape((1, 1))  # Reshape the input data to (batch_size, input_dim)
    return model.predict(input_data)[0][0]


# Define option parameters
S = 100
K = 105
r = 0.05
q = 0.02
T = 0.5
sigma = 0.2

# Generate a larger training set
n_train = 10000
S_train = np.random.normal(loc=S, scale=S*sigma*np.sqrt(T), size=n_train)
X_train = np.stack([S_train, K*np.ones(n_train), r*np.ones(n_train), q*np.ones(n_train), T*np.ones(n_train)], axis=1)
y_train = black_scholes_call_price(S_train, K, r, q, T, sigma)

# Train the neural network
model = black_scholes_nn(S, K, r, q, T, sigma)
model.fit(X_train, y_train, epochs=200, batch_size=256, verbose=0)

# Evaluate the neural network on a test example
S_test = 110
X_test = np.array([[S_test, K, r, q, T]])
call_price_nn = model.predict(X_test)[0][0]
call_price_bs = black_scholes_call_price(S_test, K, r, q, T, sigma)
print(f"The predicted call price from the neural network is {call_price_nn}")
print(f"The Black-Scholes call price is {call_price_bs}")
