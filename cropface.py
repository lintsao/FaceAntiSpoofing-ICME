#from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import transforms
from PIL import Image
import cv2
#mtcnn = MTCNN(image_size=256)

from PIL import Image

from mtcnn import MTCNN
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
def crop_image(image_path):
    detector = MTCNN() 
    img=cv2.imread(image_path)
    data=detector.detect_faces(img)
    biggest=0
    if data !=[]:
        for faces in data:
            box=faces['box']            
            # calculate the area in the image
            area = box[3]  * box[2]
            if area>biggest:
                biggest=area
                bbox=box 
        bbox[0]= 0 if bbox[0]<0 else bbox[0]
        bbox[1]= 0 if bbox[1]<0 else bbox[1]
        img=img[bbox[1]: bbox[1]+bbox[3],bbox[0]: bbox[0]+ bbox[2]] 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert from bgr to rgb
        return (True, img) 
    else:
        return (False, None)


#img = Image.open()
img_path=  "/groups/virginiakm1988/frank_dataset/casia/train/crop_frame/10_1.jpg"
status,img=crop_image(img_path)
if status:
    plt.imshow(img)
else:
    print('No facial image was detected')

# Get cropped and prewhitened image tensor
#img_cropped = mtcnn(img, save_path="./cropped_img/new.jpg")
#new = img_cropped.ToPILImage()
#new.save("./cropped_img/new.jpg")