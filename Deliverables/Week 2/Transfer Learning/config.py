"""
Data Configurations
"""

PATH_TRAINING = "/home/denizkorkmaz/Desktop/CMPE492/Transfer Learning/Example Dataset/Training"
PATH_TEST = "/home/denizkorkmaz/Desktop/CMPE492/Transfer Learning/Example Dataset/Test"

TRAINING_BATCH_SIZE = 64
TEST_BATCH_SIZE = 32

raw_data_config = {
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
    'Validation Split':0.2
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
TOTAL_EPOCHS =  INITIAL_EPOCHS + FINE_TUNE_EPOCHS
FINE_TUNE_AT = 15

"""
Evaluation Configurations
"""
SIGMOID_THRESHOLD = 0.5