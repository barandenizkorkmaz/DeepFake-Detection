import os

"""
Data Configurations
"""
NAME_DATA = "CMPE492_Deepfakedetection_Data"
PATH_TRAINING = os.path.join(os.getcwd(),"{}/Training/Training".format(NAME_DATA))
PATH_VALIDATION = os.path.join(os.getcwd(),"{}/Training/Validation".format(NAME_DATA))
PATH_TEST = os.path.join(os.getcwd(),"{}/Test".format(NAME_DATA))

TRAINING_BATCH_SIZE = 64
TEST_BATCH_SIZE = 32

raw_data_config = { # Not used anymore!
    'Batch Size':TRAINING_BATCH_SIZE,
    'Height':180,
    'Width':180,
    'Channel':3,
    'Shuffle':True,
    'Seed':123,
    'Validation Split':0.2
}

training_data_config = {
    'Batch Size':TRAINING_BATCH_SIZE,
    'Height':180,
    'Width':180,
    'Channel':3,
    'Shuffle':True,
    'Seed':123,
    'Validation Split':None
}

validation_data_config = {
    'Batch Size':TRAINING_BATCH_SIZE,
    'Height':180,
    'Width':180,
    'Channel':3,
    'Shuffle':True,
    'Seed':123,
    'Validation Split':None
}

test_data_config = {
    'Batch Size':TEST_BATCH_SIZE,
    'Height':180,
    'Width':180,
    'Channel':3,
    'Shuffle':False,
    'Seed':None,
    'Validation Split':None
}
img_shape = (training_data_config['Height'],training_data_config['Width'],training_data_config['Channel'])
NUM_FRAMES_PER_VIDEO = 50

"""
Classifier Configurations
"""
# BASE MODEL
NUMBER_OF_CLASSES = 1
BASE_LEARNING_RATE = 0.0001
INITIAL_EPOCHS = 10
VALIDATION_STEPS = 20

# FINE TUNING
FINE_TUNE_EPOCHS = 10
TOTAL_EPOCHS =  15
FINE_TUNE_AT = 15

#LSTM
NUMBER_OF_UNITS = 1024 # DIMENSIONALITY OF HIDDEN LAYERS
LSTM_EPOCHS = 20
LSTM_INPUT_SHAPE = (50,512) # (timesteps, features)

"""
Evaluation Configurations
"""
SIGMOID_THRESHOLD = 0.5

"""
CONSTANTS
"""
PATH_CNN = os.path.join(os.getcwd(), "CNN_Model")
PATH_LSTM = os.path.join(os.getcwd(), "LSTM_Model")

PATH_CNN_TRAINING_SCRIPT = os.path.join(os.getcwd(),"model/CNN/train_cnn.py")
PATH_CNN_TEST_SCRIPT = os.path.join(os.getcwd(),"model/CNN/test_cnn.py")

PATH_LSTM_TRAINING_SCRIPT = os.path.join(os.getcwd(),"model/LSTM/train_lstm.py")
PATH_LSTM_TEST_SCRIPT = os.path.join(os.getcwd(),"model/LSTM/test_lstm.py")

PATH_CSV_TRAINING = os.path.join(os.getcwd(), "features_training.csv")
PATH_CSV_VALIDATION = os.path.join(os.getcwd(), "features_validation.csv")
PATH_CSV_TEST = os.path.join(os.getcwd(), "features_test.csv")
PATH_CSV_FINAL = os.path.join(os.getcwd(), "Final_Output.csv")

"""
PLOT CONSTANTS
"""
ACCURACY_PLOT_ID = 1
CONFUSION_MATRIX_PLOT_ID = 1