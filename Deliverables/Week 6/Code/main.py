import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import config
from model.CNN import cnn_utils

if len(sys.argv)!=2:
    raise SystemExit("Please enter valid arguments: [-base] or [-finetuning]")
train_cnn_arg = sys.argv[1] # -base / -finetuning

is_cnn_model_found = cnn_utils.search_model(config.PATH_CNN)
is_lstm_model_found = cnn_utils.search_model(config.PATH_LSTM)

commands_training = [
    "python3 {} {}".format(config.PATH_CNN_TRAINING_SCRIPT,train_cnn_arg),
    "python3 {}".format(config.PATH_LSTM_TRAINING_SCRIPT)
]

commands_test = [
    "python3 {}".format(config.PATH_CNN_TEST_SCRIPT),
    "python3 {}".format(config.PATH_LSTM_TEST_SCRIPT)
]

if (not is_cnn_model_found) or (not is_lstm_model_found):
    for command in commands_training:
        print("Running: {}".format(command))
        os.system(command=command)

for command in commands_test:
    print("Running: {}".format(command))
    os.system(command=command)