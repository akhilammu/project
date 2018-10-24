import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('F:/img/tst.png',0)
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(img,cv.MORPH_OPEN,kernel, iterations = 2)
sure_bg = cv.dilate(opening,kernel,iterations=3)
cv2.imwrite('F:/img/tst2.png',sure_bg)
plt.subplot(121),plt.imshow(sure_bg, cmap = 'gray')
plt.title('noice removal'), plt.xticks([]), plt.yticks([])
plt.show()