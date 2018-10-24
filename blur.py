import cv2
import os
import glob
img_dir = "C:/users/hp/image/" # Enter Directory of all images
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
for f1 in files:
    im = cv2.imread(f1)
    img_blur_small = cv2.GaussianBlur(im, (5,5), 0)
    cv2.imwrite(f1, img_blur_small)