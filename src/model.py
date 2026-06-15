from __future__ import annotations


def build_lstm(window, n_features, lstm_units, dropout, learning_rate):
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(window, n_features))
    x = inputs
    for i, units in enumerate(lstm_units):
        x = layers.LSTM(units, return_sequences=i < len(lstm_units) - 1)(x)
        if dropout > 0:
            x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, name="return_head")(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="mse")
    return model
