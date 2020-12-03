import pathlib
import os
import sys

import config
from utils import data_handler, evaluation_utils
from model.CNN import cnn_utils

#
# Delete Previous CNN Model
#
delete_cnn_command = "rm -r {}".format(config.PATH_CNN)
os.system(command=delete_cnn_command)

train_cnn_arg = sys.argv[1] # -base / -finetuning
if train_cnn_arg.lower() == '-base':
    MODE = 0
elif train_cnn_arg.lower() == '-finetuning':
    MODE = 1
else:
    raise SystemExit("Please enter valid arguments: [-base] or [-finetuning]")

# Import Training Data
training_data_dir = pathlib.Path(config.PATH_TRAINING)
validation_data_dir = pathlib.Path(config.PATH_VALIDATION)
raw_train = data_handler.import_data(training_data_dir,config.training_data_config)
raw_validation = data_handler.import_data(validation_data_dir,config.validation_data_config)

# Format the Dataset
train = raw_train.map(data_handler.format_example)
validation = raw_validation.map(data_handler.format_example)

# Create CNN Model
base_cnn_model = cnn_utils.get_base_model()
cnn_model = cnn_utils.create_model(base_cnn_model) if MODE == 0 else cnn_utils.fine_tune_model(base_cnn_model)

# Initial CNN Model Status.
loss0,accuracy0 = cnn_model.evaluate(validation, steps = config.VALIDATION_STEPS)
print("Initial Loss = {:.2f}, Initial Accuracy = {:.2f}".format(loss0,accuracy0))

# Train Base Model (MODE == 0) OR Fine-Tuned Model (MODE == 1) & Evaluate
history = cnn_model.fit(train, epochs=config.TOTAL_EPOCHS, validation_data=validation)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
evaluation_utils.accuracy_plot((acc, loss), (val_acc, val_loss))

# Save Model
cnn_model.save(config.PATH_CNN)