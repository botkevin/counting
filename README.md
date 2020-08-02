# counting

## feature detection
A framework for feature detection. Framework supports (1)[```ORB```, ```SIFT```, and ```SURF```] with both (2)[```FLANN``` and ```brute-force```] nearest neigbor matching with a final pruning algorithm of (3)[```crosscheck``` or ```Lowe's ratio test```]. Use ```detect_organized.py``` for basic feature detection and ```display.py``` to see results. Look at ```detect_organized_t.py``` under ```/tests``` for an example

## object detection
This project's main goal is to detect multiple objects. The method is composed of:

### Image object segmentation
We use selective search to find potential objects

### Basic cutoff
Basic cutoff of potential objects that do not have enough features to be meaningful

### Matchbox reduction
Matchbox reduction using homography matricies to reduce object bounding box using homography transforms from master image to respective matchboxes

### Homography angle cutoff
Using angle measurement of homography quadrilaterals to prune matchboxes that have multiple of desired object within

### NMS
Non-Maximum supression to prune redundant/overlapping matchboxes


#### If you want to see an example of all of this in motion, check out ```tests/roi_trailmix_t.ipynb```

cv2... more info later
