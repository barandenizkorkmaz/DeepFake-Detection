import tensorflow as tf
import numpy as np
import os
import pathlib
import csv

import config

def import_data(directory,config):
  """
  Imports the data according to given configurations.
  Arguments:
    directory: Directory where the data is located.
        If `labels` is "inferred", it should contain
        subdirectories, each containing images for a class.
        Otherwise, the directory structure is ignored.
    config: [Dict] Previously determined data configurations.
    Example:
        config = {
          'Batch Size':64,
          'Height':160,
          'Width':160,
          'Channel':3,
          'Shuffle':True,
          'Seed':123,
          'Validation Split':0.2
        }
    subset: The title of subset created.
        "training": Training Data
        "validation": Validation Data
        None: Test Data
  """
  return tf.keras.preprocessing.image_dataset_from_directory(
    directory,
    batch_size=config['Batch Size'],
    image_size=(config['Height'], config['Width']),
    shuffle=config['Shuffle'],
    seed=config['Seed'],
    validation_split=config['Validation Split']
  )

def get_file_names(path):
  file_names = list()
  for root, dirs, files in os.walk(path):
    dirs.sort()
    for d in sorted(dirs):
      subdir = os.path.join(root, d)
      file_names.extend(sorted(os.listdir(subdir)))
  return file_names

def format_example(image, label):
  """ Rescales the dataset into [-1,1]. The new configuration of data must
  be predetermined.
  """
  image = tf.cast(image, tf.float32)
  image = (image/127.5) - 1
  image = tf.image.resize(image, (config.img_shape[0], config.img_shape[1]))
  return image, label

def string_to_array(features_string):
  features_stripped = features_string[1:-1]
  features_split = features_stripped.split()
  return list(map(float, features_split))

def read_csv_file(file, num_of_frames=config.NUM_FRAMES_PER_VIDEO):
  video_features = []
  labels = list()
  index = 0
  with open(file, 'r') as read_obj:
    csv_reader = csv.reader(read_obj)
    header = next(csv_reader)
    for row in csv_reader:
      if index % num_of_frames == 0:
        video_features.append([])
        labels.append(int(row[3]))
      frame = string_to_array(row[2])
      video_features[-1].append(frame)
      index += 1
  return np.array(video_features), np.array(labels)

"""
Example: manipulated_sequences_034_590_Deepfakes_c40_1
Format: {manipulated_sequences/original_sequences}_{video_id}_{Manipulation Method}_{Video Quality}_{Frame Number}.png
video_id : 034_590
"""
def get_video_name(file_name):
  video_name = ""
  video_name_split = file_name.split("_")
  tmp = []
  for element in video_name_split:
    try:
      int(element)
      tmp.append(element)
    except ValueError:
      pass
  for element in tmp:
    video_name = video_name + element + "_"
  return video_name[:-1]


def get_video_names(file_names, number_of_frames=config.NUM_FRAMES_PER_VIDEO):
  video_names = []
  index = 0
  for file_name in file_names:
    if index % number_of_frames == 0:
      video_name = get_video_name(file_name)
      video_names.append(video_name)
    index += 1
  return video_names