import os

current_path = os.getcwd()
ALL_DATASETS = ['original', 'DeepFakeDetection_original', 'Deepfakes',
                'DeepFakeDetection', 'Face2Face', 'FaceShifter', 'FaceSwap',
                'NeuralTextures']

selected_datasets = ['original','Deepfakes','Face2Face','FaceShifter','FaceSwap','NeuralTextures']

"""
Dataset Options:
{original_youtube_videos,original_youtube_videos_info,original,DeepFakeDetection_original,Deepfakes,DeepFakeDetection,Face2Face,FaceShifter,FaceSwap,NeuralTextures,all}
"""

for dataset in selected_datasets:
    command = "python3 faceforensics_download_v4.py {} -d {} -c c40 -t videos".format(current_path,dataset)
    os.system(command)

#EXAMPLES:
#os.system('python3 faceforensics_download_v4.py ~/PycharmProjects/FaceForensics++/ -d original -c raw -t videos -n 4')
#os.system('python3 faceforensics_download_v4.py ~/PycharmProjects/FaceForensics++/ -d Deepfakes -c raw -t videos -n 4')
#os.system('python3 faceforensics_download_v4.py ~/PycharmProjects/FaceForensics++/ -d FaceSwap -c raw -t videos -n 4')
#os.system('python3 faceforensics_download_v4.py ~/PycharmProjects/FaceForensics++/ -d Face2Face -c raw -t videos -n 4')