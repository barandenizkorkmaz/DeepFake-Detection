import tensorflow as tf

import config

def get_lstm_model(number_of_units=config.NUMBER_OF_UNITS):
    lstm_layer = tf.keras.layers.LSTM(number_of_units, input_shape=config.LSTM_INPUT_SHAPE)
    dense_layer_1 = tf.keras.layers.Dense(1024)
    dense_layer_2 = tf.keras.layers.Dense(32)
    dense_layer_3 = tf.keras.layers.Dense(1)
    model = tf.keras.Sequential([
        lstm_layer,
        dense_layer_1,
        dense_layer_2,
        dense_layer_3
    ])
    model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.optimizers.Adam(lr=config.BASE_LEARNING_RATE / 10),
                  metrics=[tf.metrics.BinaryAccuracy(threshold=0.0, name='accuracy')])
    return model