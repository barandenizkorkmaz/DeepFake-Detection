import pathlib
import tensorflow as tf

import config
from utils import data_handler, evaluation_utils

#
# Import Previously Trained CNN.
#
cnn_model = tf.keras.models.load_model(config.PATH_CNN)

#
# Remove output layers!
#
feature_cnn_model = tf.keras.models.Sequential(cnn_model.layers[:-2])

# Import & Format Test Data
test_data_dir = pathlib.Path(config.PATH_TEST)
raw_test = data_handler.import_data(test_data_dir, config.test_data_config)
cnn_test = raw_test.map(data_handler.format_example)

# Read the names of files/videos in ascending order by name
# same as they are imported in the dataset.
file_names = data_handler.get_file_names(config.PATH_TEST)
output_feature_cnn = feature_cnn_model.predict(cnn_test)

# Write Features of Frames to CSV
evaluation_utils.write_features_to_csv(config.PATH_CSV_TEST, (file_names, output_feature_cnn),
                                       ['#', 'File Name', 'Features', 'Labels'])