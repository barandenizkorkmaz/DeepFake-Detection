import tensorflow as tf
import pathlib
import os

import config
from utils import data_handler, evaluation_utils
from model.LSTM import lstm_utils

#
# Delete Previous LSTM Model
#
delete_lstm_command = "rm -r {}".format(config.PATH_LSTM)
delete_csv_command = "rm {}".format(config.PATH_CSV_FINAL)
# Delete previous outputs.
delete_csv_training_command = "rm {}".format(config.PATH_CSV_TRAINING)
delete_csv_test_command = "rm {}".format(config.PATH_CSV_TEST)
os.system(command=delete_lstm_command)
os.system(command=delete_csv_command)
os.system(command=delete_csv_training_command)
os.system(command=delete_csv_test_command)

cnn_model = tf.keras.models.load_model(config.PATH_CNN)

#
# Remove output layers!
#
feature_cnn_model = tf.keras.models.Sequential(cnn_model.layers[:-2])

# Import Training Data
training_data_dir = pathlib.Path(config.PATH_TRAINING)
validation_data_dir = pathlib.Path(config.PATH_VALIDATION)
raw_train = data_handler.import_data(training_data_dir,config.training_data_config)
raw_validation = data_handler.import_data(validation_data_dir,config.validation_data_config)

# Format the Dataset
cnn_training = raw_train.map(data_handler.format_example)
cnn_validation = raw_validation.map(data_handler.format_example)

# Read the names of files/videos in ascending order by name
# same as they are imported in the dataset.
file_names_training = data_handler.get_file_names(config.PATH_TRAINING)
file_names_validation = data_handler.get_file_names(config.PATH_VALIDATION)

# Feed Entire Training Data into Pretrained CNN
# Input: Video Frames of Size (180,180,3)
# Output: Features of Dim 512.
output_features_training_cnn = feature_cnn_model.predict(cnn_training)
output_features_validation_cnn = feature_cnn_model.predict(cnn_validation)

# Write Features of Frames to CSV
evaluation_utils.write_features_to_csv(config.PATH_CSV_TRAINING, (file_names_training, output_features_training_cnn), ['#', 'File Name', 'Features', 'Labels'])
evaluation_utils.write_features_to_csv(config.PATH_CSV_VALIDATION, (file_names_validation,output_features_validation_cnn), ['#', 'File Name', 'Features', 'Labels'])

# Get the training data of LSTM that is extracted by CNN, and actual labels of videos.
lstm_x_training, lstm_y_actual_training = data_handler.read_csv_file(config.PATH_CSV_TRAINING)
lstm_x_validation, lstm_y_actual_validation = data_handler.read_csv_file(config.PATH_CSV_VALIDATION)


# Create LSTM Model
lstm_model = lstm_utils.get_lstm_model()

# Train LSTM Model.
history = lstm_model.fit(x=lstm_x_training, y=lstm_y_actual_training, epochs=config.LSTM_EPOCHS, validation_data=(lstm_x_validation,lstm_y_actual_validation), batch_size=32)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
evaluation_utils.accuracy_plot((acc, loss), (val_acc, val_loss))

# Save LSTM Model
lstm_model.save(config.PATH_LSTM)