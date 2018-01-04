import os
import subprocess
def set_yolo_weights():
    if os.path.exists('./bin/yolo.weights'):

        print("yolo.weights is exists")

    else:
        cmd1 = ['wget', 'https://pjreddie.com/media/files/yolo.weights']


        os.mkdir('./bin')
        subprocess.run(cmd1)
        os.rename('./yolo.weights', './bin/yolo.weights')
