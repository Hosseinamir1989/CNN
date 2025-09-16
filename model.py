from tensorflow.keras.models             import Sequential, load_model
from tensorflow.keras.layers             import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.layers import Dropout



def build_cnn_model(window_size, n_features):
    m = Sequential([
        Conv1D(64, 3, activation="relu", padding="same", input_shape=(window_size, n_features)),
        #Conv1D(128, 3, activation="relu", padding="same"),
        MaxPooling1D(2),
        Flatten(),
        #Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        #Dropout(0.2),
        Dense(1)
    ])
    m.compile(loss="mean_squared_error", optimizer="adam")
    return m
