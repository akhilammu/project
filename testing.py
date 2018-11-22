import cv2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import skimage, os
from skimage.morphology import (ball, disk, dilation, binary_erosion,
remove_small_objects, erosion, closing, reconstruction, binary_closing)
from skimage.measure import (label,regionprops, perimeter)
from skimage.morphology import (binary_dilation, binary_opening)
from skimage.filters import (roberts, sobel)
from skimage.segmentation import clear_border
import glob
import csv

img=img = cv2.imread('project/img/test/test2.png', 0)
kernel = np.ones((5,5), np.uint8)
img = cv2.dilate(img, kernel, iterations=1)
(thresh, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY |
cv2.THRESH_OTSU)
thresh = 127
im_bw = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
cleared = clear_border(im_bw)

label_image = label(cleared)


areas = [r.area for r in regionprops(label_image)]
areas.sort()
if len(areas) > 2:
    for region in regionprops(label_image):
        if region.area < areas[-2]:
            for coordinates in region.coords:
                label_image[coordinates[0], coordinates[1]] = 0
binary = label_image > 0



selem = disk(2)
binary = binary_erosion(binary, selem)

selem = disk(10)
binary = binary_closing(binary, selem)

edges = roberts(binary)
binary = ndi.binary_fill_holes(edges)

get_high_vals = binary == 0
img[get_high_vals] = 0

image, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


numOfContours = len(contours)   #number of contours

area = []
perimeter=[]
count = 0
for count in range(numOfContours) :
    cv2.drawContours(img, contours, -1, (20,255,60), 1)  #draw contours
    cnt = contours[count]
    area.append(cv2.contourArea(cnt))
    peri = cv2.arcLength(cnt,True)
    perimeter.append(peri)
    #print(area)
    
    
    count+=1
    #print(contours)

#print(numOfContours)    
if len(area)==0:
    print("")
else:
        
    a=max(area)
    #print(x)
    print("area:",a)   #gives the largest area
    for i in range(numOfContours) :
        if area[i]==a:
            k=i
    if a<30:
        e=1
    else:
        cnt = contours[k]
        ellipse = cv2.fitEllipse(cnt)
        (center,axes,orientation) = ellipse
        majoraxis_length = max(axes)
        minoraxis_length = min(axes)
        e=minoraxis_length/majoraxis_length
    p=perimeter[k]
    print("perimeter:",p)
    print("Eccentricity:",e)
    csvTitle = [['Area', 'Perimeter','Eccentricity']]
    csvData = []
    csvData.append([a, p, e])
    with open('new.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvTitle)
        writer.writerows(csvData)
    d=pd.read_csv("new.csv")
    d_new=pd.read_csv("new.csv",na_values=['?'])

    d_new.dropna(inplace=True)
    data=pd.read_csv("features.csv")
    data_new=pd.read_csv("features.csv",na_values=['?'])

    data_new.dropna(inplace=True)
    predictions=data_new['prediction']
    features_raw = data_new[['Area','Perimeter','Eccentricity']]
    X_train=features_raw
    y_train=predict_class
    X_test=d_new[['Area','Perimeter','Eccentricity']]
    print ("Training set has {} samples." .format(X_train.shape[0]))
    print ("Testing set has {} samples." .format(X_test.shape[0]))
    import sklearn
    from sklearn import svm

    C = 1.0
    svc = svm.SVC(kernel='linear',C=C,gamma=2)
    svc.fit(X_train, y_train)
    from sklearn.metrics import fbeta_score
    print(X_test)
    predictions_test = svc.predict(X_test)
    print(predictions_test)
    if(predictions_test[0]==1):
        print("Cancer detected")
    else:
        print("Normal")

