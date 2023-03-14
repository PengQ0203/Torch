import os
import json
import cv2
from background_segmentation import background_segmentation


if __name__ == '__main__':
    #setting up parameters
    #choose learning rate and threshold values manually.
    testFile = "./param.json"
    video_path = cv2.VideoCapture(r"../../../video/202-233/FormatFactoryPart3.mp4")

    if os.path.exists(testFile):
        with open(testFile, 'r') as f:
            jsonObj = json.load(f)
        alpha= jsonObj["learning_rate"]
        T = jsonObj["threshold"]

    object1=background_segmentation(video_path,alpha,T)
    object1.update_parameter()

