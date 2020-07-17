import sys
sys.path.append("../")
import roi
import display

import skimage.io
import cv2
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
import time

image = skimage.io.imread("../images/find_the_screen.jpg")
image = resize(image, (image.shape[0] // 10, image.shape[1] // 10))
boxes = roi.s_search(image)
display.just_boxes(boxes,image)
# for box in boxes:
#     start_point = box[:2] 
#     end_point = box[2:]
#     color = (255, 0, 0) 
#     thickness = 1
#     image = cv2.rectangle(image, start_point, end_point, color, thickness)
# plt.imshow(image)
# plt.show()
