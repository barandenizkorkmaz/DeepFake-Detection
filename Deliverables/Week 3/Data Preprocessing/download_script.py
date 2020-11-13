import os

current_path = os.getcwd()
ALL_DATASETS = ['original', 'DeepFakeDetection_original', 'Deepfakes',
                'DeepFakeDetection', 'Face2Face', 'FaceShifter', 'FaceSwap',
                'NeuralTextures']

set_A = ['original','Deepfakes','Face2Face','FaceShifter','FaceSwap','NeuralTextures']
set_B = ['DeepFakeDetection_original','DeepFakeDetection']

"""
Dataset Options:
{original_youtube_videos,original_youtube_videos_info,original,DeepFakeDetection_original,Deepfakes,DeepFakeDetection,Face2Face,FaceShifter,FaceSwap,NeuralTextures,all}
"""

for dataset in set_A:
    command = "python3 main.py {} -d {} -c c40 -t videos".format(current_path,dataset)
    print(command)
    #os.system(command)

# EXAMPLES:
#os.system('python3 main.py ~/PycharmProjects/FaceForensics++/ -d original -c raw -t videos -n 4')
#os.system('python3 main.py ~/PycharmProjects/FaceForensics++/ -d Deepfakes -c raw -t videos -n 4')
#os.system('python3 main.py ~/PycharmProjects/FaceForensics++/ -d FaceSwap -c raw -t videos -n 4')
#os.system('python3 main.py ~/PycharmProjects/FaceForensics++/ -d Face2Face -c raw -t videos -n 4')
