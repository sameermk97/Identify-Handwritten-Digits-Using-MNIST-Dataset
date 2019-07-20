import cv2
from keras.models import load_model
import numpy as np

def testimg(filepath):
    IMG_SIZE=28
    img_array=cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
    return new_array.reshape(-1,IMG_SIZE,IMG_SIZE,1)



model=load_model('C:/Users/Kulkarni/Downloads/Computer-Vision-with-Python/DATA/mnist_model.h5')

prediction=model.predict([testimg('C:/Users/Kulkarni/Desktop/zero.png')])
print("The detected number is\n")
print(np.argmax(prediction))

