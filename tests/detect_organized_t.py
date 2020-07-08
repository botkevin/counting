import sys
sys.path.append("../")
import cv2
import detect_organized
import skimage.io
from skimage.color import rgb2gray
from skimage.transform import rescale, resize
import imutils
import display

# img1 = skimage.io.imread("../images/screen.jpg")
img1 = cv2.imread('../images/screen.jpg',0)
img1 = imutils.resize(img1, width=500)

# img2 = skimage.io.imread("../images/find_the_screen.jpg")
img2 = cv2.imread('../images/find_the_screen.jpg',0)
img2 = imutils.resize(img2, width=500)

method = "surf"
dec = detect_organized.detector(method)
kp1, des1 = detect_organized.detect(dec, img1)
kp2, des2 = detect_organized.detect(dec, img2)

matches = detect_organized.match(des1, des2, "FLANN", method)
good = detect_organized.ratio(matches)

display.show(kp1, kp2, img1, img2, good, flag = 0)
display.show(kp1, kp2, img1, img2, good, flag = 1)
display.show(kp1, kp2, img1, img2, good, flag = 2)
