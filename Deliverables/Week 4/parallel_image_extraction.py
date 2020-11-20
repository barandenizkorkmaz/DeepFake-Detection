import cv2
import os
import pathlib
import multiprocessing

current_directory = os.getcwd()

root_directory = ['manipulated_sequences','original_sequences'] # Manually configured!
directories_to_scan = []

def process_directory(dir):
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file[-4:] == ".mp4":
                cur_path = pathlib.Path(root)
                parent_path = cur_path.parent
                target_path = os.path.join(parent_path, 'images')
                if not os.path.exists(target_path):
                    os.makedirs(target_path)
                file_path = os.path.join(cur_path, file)
                target_file_path = os.path.join(target_path, file[:-4])
                print("Processing: "+ file_path + " To: "+target_file_path)
                video_to_frames(file_path, target_file_path)

def video_to_frames(video, path_output_dir):
    # extract frames from a video and save to directory as 'x.png' where
    # x is the frame index
    if not os.path.exists(path_output_dir):
        os.makedirs(path_output_dir)
        #print("Processing: " + path_output_dir + "\n")
    vidcap = cv2.VideoCapture(video)
    count = 0
    while vidcap.isOpened():
        success, image = vidcap.read()
        if success:
            cv2.imwrite(os.path.join(path_output_dir, '%d.png') % count, image)
            count += 1
        else:
            break
    cv2.destroyAllWindows()
    vidcap.release()

for dir in root_directory:
    subdir = os.path.join(current_directory,dir)
    for root,dirs,files in os.walk(os.path.join(current_directory,dir)):
        for subdirectory in dirs:
            next_directory = os.path.join(subdir, subdirectory)
            directories_to_scan.append(next_directory)
        break

for dir in directories_to_scan:
    p = multiprocessing.Process(target=process_directory,args=(dir,))
    p.start()