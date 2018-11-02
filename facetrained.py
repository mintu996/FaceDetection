import os
import cv2
import numpy as np
from PIL import Image
recognizer=cv2.face.LBPHFaceRecognizer_create();
path='dataset'
def getImageWithID(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    Ids=[]
    for i in imagePaths:
        faceImg=Image.open(i).convert('L');
        faceNp=np.array(faceImg,'uint8')
        ID=int(os.path.split(i)[-1].split('_')[1])
        faces.append(faceNp)
        Ids.append(ID)
        cv2.imshow("training",faceNp)
        cv2.waitKey(10)
    return np.array(Ids),faces

Ids,faces=getImageWithID(path)
recognizer.train(faces,Ids)
recognizer.save("Recognition/trainingdata.yml")
cv2.destroyAllWindows()