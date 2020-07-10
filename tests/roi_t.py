import sys
sys.path.append("../")
import detect_organized as det
import roi
import display

import cv2
import imutils
from matplotlib import pyplot as plt

img1 = cv2.imread('../images/screen.jpg',0)
img1 = imutils.resize(img1, width=500)
img2 = cv2.imread('../images/find_the_screen.jpg',0)
img2 = imutils.resize(img2, width=500)
method = "surf"
dec = det.detector(method)


def test_mask():
    kp1, des1 = det.detect(dec, img1)
    print (img2.shape)
    rect = (0, 100, 100, 200)
    mask = roi.make_mask(img2.shape, rect)
    plt.imshow(mask),plt.show()
    kp2, des2 = det.detect(dec, img2, mask)
    print(len(kp1) ,"|", len(kp2))
    matches = det.match(des1, des2, "FLANN", method)
    good = det.ratio(matches)

    display.show(kp1, kp2, img1, img2, good, flag = 0)

test_mask()