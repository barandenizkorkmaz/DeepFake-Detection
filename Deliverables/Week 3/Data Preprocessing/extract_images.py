import cv2
import os
import pathlib

current_directory = os.getcwd()

root_directory = ['original_sequences','manipulated_sequences',]

def video_to_frames(video, path_output_dir):
    # extract frames from a video and save to directory as 'x.png' where
    # x is the frame index
    if not os.path.exists(path_output_dir):
        os.makedirs(path_output_dir)
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
    if dir == 'original_sequences':
        for root, dirs, files in os.walk(os.path.join(current_directory, dir)):
            for file in files:
                if file[-4:] == ".mp4":
                    cur_path = pathlib.Path(root)
                    parent_path = cur_path.parent
                    target_path = os.path.join(parent_path, 'images')
                    if not os.path.exists(target_path):
                        os.makedirs(target_path)
                    file_path = os.path.join(cur_path,file)
                    target_file_path = os.path.join(target_path,file[:-4])
                    #print("Processing: "+ file_path + " To: "+target_file_path + "\n")
                    video_to_frames(file_path,target_file_path)

    else:
        for root,dirs,files in os.walk(os.path.join(current_directory,dir)):
            for file in files:
                if file[-4:] == ".mp4":
                    cur_path = pathlib.Path(root)
                    parent_path = cur_path.parent
                    target_path = os.path.join(parent_path,'images')
                    if not os.path.exists(target_path):
                        os.makedirs(target_path)
                    file_path = os.path.join(cur_path,file)
                    target_file_path = os.path.join(target_path,file[:-4])
                    #print("Processing: "+ file_path + " To: "+target_file_path + "\n")
                    video_to_frames(file_path,target_file_path)

#video_to_frames('/home/denizkorkmaz/PycharmProjects/FaceForensics++/manipulated_sequences/Deepfakes/raw/videos/585_599.mp4', '/home/denizkorkmaz/PycharmProjects/FaceForensics++/manipulated_sequences/Deepfakes/raw/images/585_599/')
