import multiprocessing
import os
import pathlib
import shutil
import sys

import config

current_directory = os.getcwd()
data_dir = os.path.join(current_directory,config.NAME_DATA)
root_directory = ['original_sequences','manipulated_sequences']
directories_to_scan = []

def create_dirs():
    training_data_path = os.path.join(data_dir, 'Training')
    test_data_path = os.path.join(data_dir, 'Test')
    training_training_data_path = os.path.join(training_data_path, 'Training')
    training_validation_data_path = os.path.join(training_data_path, 'Validation')
    training_training_fake_data_path = os.path.join(training_training_data_path, 'Fake')
    training_training_real_data_path = os.path.join(training_training_data_path, 'Real')
    training_validation_fake_data_path = os.path.join(training_validation_data_path, 'Fake')
    training_validation_real_data_path = os.path.join(training_validation_data_path, 'Real')
    test_fake_data_path = os.path.join(test_data_path, 'Fake')
    test_real_data_path = os.path.join(test_data_path, 'Real')
    
    os.makedirs(training_training_fake_data_path)
    os.makedirs(training_training_real_data_path)
    os.makedirs(training_validation_fake_data_path)
    os.makedirs(training_validation_real_data_path)
    os.makedirs(test_fake_data_path)
    os.makedirs(test_real_data_path)


def copy_images(source_file_path_template, indices, is_fake, is_training, is_validation):
    for index in indices:
        source_file_full_path = source_file_path_template + str(index) + ".png"
        target_path = os.path.join(data_dir, 'Training') if is_training else os.path.join(data_dir, 'Test')
        if is_training:
            target_path = os.path.join(target_path, 'Training') if not is_validation else os.path.join(target_path, 'Validation')
        if is_fake:
            #print("Processing: "+ source_file_full_path + " To: "+ os.path.join(target_path, "Fake"))
            shutil.copy(source_file_full_path, "{}/Fake".format(target_path))
        else:
            #print("Processing: "+ source_file_full_path + " To: "+ os.path.join(target_path, "Real"))
            shutil.copy(source_file_full_path, "{}/Real".format(target_path))


def get_indices(num_of_frames, count=50):
    indices = []
    stride = int(num_of_frames/count)
    index = 1
    while len(indices) != count:
        indices.append(index)
        index = (index+stride) % num_of_frames
    return indices


def format_file_name(file_name):
    index = len(file_name)-1
    for element in file_name[::-1]:
        if element == '_':
            return index
        index = index - 1


def process_directory(dir):
    for root, dirs, files in os.walk(dir):
        for curr_dir in dirs:
            if curr_dir != 'images':
                continue
            images_path = os.path.join(pathlib.Path(root),curr_dir)
            is_fake = True if "manipulated_sequences" in images_path else False
            for images_root, images_dir, images_files in os.walk(images_path):
                num_of_videos = len(images_dir)
                cut_off = int(round(num_of_videos*0.8))
                for video_index,sub_dir in enumerate(images_dir):
                    current_frames_dir = os.path.join(images_path, sub_dir)
                    is_training = True if video_index < cut_off else False
                    cut_off_training = int(round(cut_off * 0.8))
                    is_validation = True if video_index >= cut_off_training else False
                    print("Processing: {}".format(current_frames_dir))
                    process_current_frames_dir(current_frames_dir, is_fake, is_training, is_validation)


def process_current_frames_dir(dir, is_fake, is_training, is_validation):
    for root, dirs, files in os.walk(dir):
        files_length = len(files)
        if files_length < 50 :
            continue
        name_template = ""
        for file in files:
            index = format_file_name(file)
            name_template = file[:index+1]
            break
        source_file_path_template = dir + "/" + name_template
        indices = get_indices(files_length)
        copy_images(source_file_path_template, indices, is_fake, is_training, is_validation)


for dir in root_directory:
    subdir = os.path.join(current_directory,dir)
    for root,dirs,files in os.walk(os.path.join(current_directory,dir)):
        for subdirectory in dirs:
            next_directory = os.path.join(subdir, subdirectory)
            directories_to_scan.append(next_directory)
        break

if not os.path.exists(data_dir):
    create_dirs()
else:
    answer = input("Already found: {}\nWould you like to overwrite?\n[Y]es | [N]o\n")

    if answer.lower() == "n":
        sys.exit(0)
    else:
        command = "rm -r {}".format(data_dir)
        os.system(command=command)
        create_dirs()

for dir in directories_to_scan:
    p = multiprocessing.Process(target=process_directory, args=(dir,))
    p.start()

