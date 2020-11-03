import tensorflow as tf
import pathlib

import evaluation_utils
import data_handler
import config
import cnn_utils

is_model_found = cnn_utils.search_model()

if is_model_found:
    print("Pretrained model has been found!")
    # Import Model
    model = tf.keras.models.load_model("my_model")
    # Import Test Data
    test_data_dir = pathlib.Path(config.PATH_TEST)
    raw_test = data_handler.import_data(test_data_dir, config.test_data_config, None)
    test = raw_test.map(data_handler.format_example)
    history = model.evaluate(test)
    file_names = data_handler.get_file_names(config.PATH_TEST)
    # Predict
    class_probabilities = model.predict(test)
    class_probabilities = tf.nn.sigmoid(class_probabilities)
    y_predicted = tf.where(class_probabilities < config.SIGMOID_THRESHOLD, 0, 1)
    # Get Actual Class Labels
    y_actual = list()
    for image,label in test:
        current_labels = label.numpy()
        y_actual.extend(current_labels)
    # Output Formatting
    class_probabilities = [item for sublist in class_probabilities.numpy() for item in sublist]
    y_predicted = [item for sublist in y_predicted.numpy() for item in sublist]
    # Accuracy
    evaluation_utils.plot_confusion_matrix(y_actual,y_predicted)
    evaluation_utils.evaluation(y_actual,y_predicted)
    # Output
    evaluation_utils.write_to_csv((file_names, class_probabilities, y_predicted),['#', 'File Name', 'Probability', 'Classification'])
    raise SystemExit("The execution has successfully finished!")


data_dir = pathlib.Path(config.PATH_TRAINING)
raw_train = data_handler.import_data(data_dir,config.raw_data_config,"training")
raw_validation = data_handler.import_data(data_dir,config.raw_data_config,"validation")

# Format the Dataset
train = raw_train.map(data_handler.format_example)
validation = raw_validation.map(data_handler.format_example)

# Create Model
base_model = cnn_utils.get_base_model()
model = cnn_utils.create_model(base_model)
print("Number of layers in the model: ", len(model.layers))

# Initial model status.
loss0,accuracy0 = model.evaluate(validation, steps = config.VALIDATION_STEPS)
print("Initial Loss = {:.2f}, Initial Accuracy = {:.2f}".format(loss0,accuracy0))

history = model.fit(train,epochs=config.INITIAL_EPOCHS,validation_data=validation)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
evaluation_utils.accuracy_plot((acc,loss),(val_acc,val_loss))


# Enhance the Model Further: Fine Tuning
model = cnn_utils.fine_tune_model(base_model)

history_fine = model.fit(train,epochs=config.TOTAL_EPOCHS,initial_epoch=history.epoch[-1],validation_data=validation)
model.save("my_model")

# Accuracy Plots
acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']
loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']
evaluation_utils.accuracy_plot((acc,loss),(val_acc,val_loss))

#
# PREDICTIONS ON TEST DATA
#

# Import & Format Test Data
test_data_dir = pathlib.Path(config.PATH_TEST)
raw_test = data_handler.import_data(test_data_dir,config.test_data_config,None)
test = raw_test.map(data_handler.format_example)

history = model.evaluate(test)

file_names = data_handler.get_file_names(config.PATH_TEST)
class_probabilities = model.predict(test)
class_probabilities = tf.nn.sigmoid(class_probabilities)
y_predicted = tf.where(class_probabilities < config.SIGMOID_THRESHOLD,0,1)
# Get Actual Class Labels
y_actual = list()
for image,label in test:
    current_labels = label.numpy()
    y_actual.extend(current_labels)

# Output Formatting
class_probabilities = [item for sublist in class_probabilities.numpy() for item in sublist]
y_predicted = [item for sublist in y_predicted.numpy() for item in sublist]
# Accuracy
evaluation_utils.plot_confusion_matrix(y_actual,y_predicted)
evaluation_utils.evaluation(y_actual,y_predicted)

# Output
evaluation_utils.write_to_csv((file_names,class_probabilities,y_predicted),['#','File Name','Probability','Classification'])