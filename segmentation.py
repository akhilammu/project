
import cv2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import skimage, os
from skimage.morphology import (ball, disk, dilation, binary_erosion,
remove_small_objects, erosion, closing, reconstruction, binary_closing)
from skimage.measure import (label,regionprops, perimeter)
from skimage.morphology import (binary_dilation, binary_opening)
from skimage.filters import (roberts, sobel)
from skimage import (measure, feature)
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pydicom
import scipy.misc
import numpy as np

from __future__ import print_function
from skimage.feature import peak_local_max
from skimage.morphology import watershed

import argparse
img = cv2.imread('project/img/dil/test112.png',0)
img = cv2.getRectSubPix(img, (320, 220), (150, 170))
dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
rows, cols = img.shape
crow,ccol = rows/2 , cols/2
crow=int(crow)
ccol=int(ccol)
# create a mask first, center square is 1, remaining all zeros
mask = np.zeros((rows,cols,2),np.uint8)
mask[crow-160:crow+160, ccol-160:ccol+160] = 1

# apply mask and inverse DFT
#fshift = dft_shift*mask
#f_ishift = np.fft.ifftshift(fshift)
#img_back = cv2.idft(f_ishift)

#img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
#cv2.imwrite('F:/img/tst1.png',img_back)
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
#plt.title('Enhanced'), plt.xticks([]), plt.yticks([])

plt.show()

(thresh, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY |
cv2.THRESH_OTSU)
thresh = 127
im_bw = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]

plot = True
if plot == True:
    f, plots = plt.subplots(9, 1, figsize=(5, 40))


if plot == True:
    plots[0].axis('off')
    plots[0].imshow(im_bw, cmap=plt.cm.bone)

cleared = clear_border(im_bw)
if plot == True:
    plots[1].axis('off')
    plots[1].imshow(cleared, cmap=plt.cm.bone)

label_image = label(cleared)
if plot == True:
    plots[2].axis('off')
    plots[2].imshow(label_image, cmap=plt.cm.bone)



areas = [r.area for r in regionprops(label_image)]
areas.sort()
if len(areas) > 2:
    for region in regionprops(label_image):
        if region.area < areas[-2]:
            for coordinates in region.coords:
                label_image[coordinates[0], coordinates[1]] = 0
binary = label_image > 0


if plot == True:
    plots[3].axis('off')
    plots[3].imshow(binary, cmap=plt.cm.bone)

#cv2.imwrite('F:/s5/proj/lung cancer/img/bin.png',binary)
selem = disk(2)
binary = binary_erosion(binary, selem)
if plot == True:
    plots[4].axis('off')
    plots[4].imshow(binary, cmap=plt.cm.bone)
selem = disk(10)
binary = binary_closing(binary, selem)
if plot == True:
    plots[5].axis('off')
    plots[5].imshow(binary, cmap=plt.cm.bone)

edges = roberts(binary)
binary = ndi.binary_fill_holes(edges)
if plot == True:
    plots[6].axis('off')
    plots[6].imshow(binary, cmap=plt.cm.bone)
#cv2.imwrite('F:/s5/proj/lung cancer/img/bin.png',binary)
get_high_vals = binary == 0
img[get_high_vals] = 0
if plot == True:
    plots[7].axis('off')
    plots[7].imshow(img, cmap=plt.cm.bone)
cv2.imwrite('project//img/nodule/test112.png',img)    
#img = cv2.getRectSubPix(img, (320, 220), (140, 170))   

#contour drawing test code
image, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#img = cv2.drawContours(image, contours, -1, (0,0, 255), 1)

numOfContours = len(contours)   #number of contours

area = []
count = 0
for count in range(numOfContours) :
    cv2.drawContours(img, contours, -1, (20,255,60), 1)  #draw contours
    cnt = contours[count]
    area.append(cv2.contourArea(cnt))
    #print(area)
    
    
    count+=1
    #print(contours)

print(numOfContours)    
print(max(area))   #gives the largest area
print(area)


#for contour in contours:
#    original = cv2.boundingRect(contour)
#    cv2.rectangle(img, (original[0], original[1]), (original[0]+original[2], original[1] + original[3]), (150,20,255), 2)
#    print(cv2.contourArea(original))

cv2.imwrite('project//img/contour/test112.png',img)
if plot == True:
    plots[8].axis('off')
    plots[8].imshow(img, cmap=plt.cm.bone)
