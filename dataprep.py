import os
import cv2

path = './imagenet50k/test/'
ll = os.listdir(path)

j=0

for i in range(len(ll)):
    j=j+1
    print(j)
    image_path = path + ll[i]
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    cv2.imwrite('./imagenet50k/test1/' + str(i) + '.jpg', img)
    