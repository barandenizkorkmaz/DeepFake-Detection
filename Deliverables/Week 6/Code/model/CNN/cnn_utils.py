import tensorflow as tf
import config
import os

def get_base_model():
    return tf.keras.applications.VGG19(input_shape=config.img_shape,
                                      include_top=False,
                                      weights='imagenet')

def create_model(base_model):
    base_model.trainable = False
    flatten = tf.keras.layers.Flatten()
    dense_layer_1 = tf.keras.layers.Dense(4096)
    dense_layer_2 = tf.keras.layers.Dense(512)
    dense_layer_3 = tf.keras.layers.Dense(64)
    prediction_layer = tf.keras.layers.Dense(1)
    model = tf.keras.Sequential([
        base_model,
        flatten,
        dense_layer_1,
        dense_layer_2,
        dense_layer_3,
        prediction_layer
    ])
    model.compile(optimizer=tf.optimizers.Adam(lr=config.BASE_LEARNING_RATE),
                  loss=tf.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[tf.metrics.BinaryAccuracy(threshold=0.0,name='accuracy')])
    return model

def fine_tune_model(base_model):
    base_model.trainable = True
    print("Number of layers in the base model: ", len(base_model.layers))
    for layer in base_model.layers[:config.FINE_TUNE_AT]:
        layer.trainable = False
    flatten = tf.keras.layers.Flatten()
    dense_layer_1 = tf.keras.layers.Dense(4096)
    dense_layer_2 = tf.keras.layers.Dense(512)
    dense_layer_3 = tf.keras.layers.Dense(64)
    prediction_layer = tf.keras.layers.Dense(1)
    model = tf.keras.Sequential([
        base_model,
        flatten,
        dense_layer_1,
        dense_layer_2,
        dense_layer_3,
        prediction_layer
    ])
    model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.optimizers.Adam(lr=config.BASE_LEARNING_RATE / 10),
                  metrics=[tf.metrics.BinaryAccuracy(threshold=0.0,name='accuracy')])
    return model

def search_model(model_path):
    return os.path.isdir(model_path)