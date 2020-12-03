import tensorflow as tf

import config
from utils import data_handler, evaluation_utils

#
# Import Previously Trained LSTM.
#
lstm_model = tf.keras.models.load_model(config.PATH_LSTM)

# Read the names of files/videos in ascending order by name
# same as they are imported in the dataset.
file_names = data_handler.get_file_names(config.PATH_TEST)
video_names = data_handler.get_video_names(file_names)

# Get the test data of LSTM that is extracted by CNN, and actual labels of videos.
lstm_x, lstm_y_actual = data_handler.read_csv_file(config.PATH_CSV_TEST)

# Predict
class_probabilities = lstm_model.predict(lstm_x)
class_probabilities = tf.nn.sigmoid(class_probabilities)
y_predicted = tf.where(class_probabilities < config.SIGMOID_THRESHOLD, 0, 1)

# Output Formatting
class_probabilities = [item for sublist in class_probabilities.numpy() for item in sublist]
y_predicted = [item for sublist in y_predicted.numpy() for item in sublist]

# Accuracy
evaluation_utils.plot_confusion_matrix(lstm_y_actual, y_predicted)
evaluation_utils.evaluation(lstm_y_actual, y_predicted)

# Output
evaluation_utils.write_predictions_to_csv(config.PATH_CSV_FINAL, (video_names, y_predicted, lstm_y_actual), ['#', 'File Name', 'Predicted Label', 'Actual Label'])