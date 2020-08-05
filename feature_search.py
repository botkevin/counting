import detect_organized as det
import roi
import display
import score
import trim

import cv2
import imutils
import skimage
from matplotlib import pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for image search')
    parser.add_argument('-m','master', 
            help="filename of source image to search for")
    parser.add_argument('-c','child', 
            help="filename of image to find the source image in")
    parser.add_argument('-md','--method', 
            help="method to use: surf, sift, or orb")
    parser.add_argument('-s','--show', 
            help="toggle showing images",
            action="store_true")
    parser.add_argument('-mb','--matchboxes',
            help="toggle matchboxes image generation",
            action="store_true")
    parser.add_argument('-sc','--score-cutoff', type=int,
            help="cutoff on matchbox score, default=100")
    parser.add_argument('-ac','--angle-cutoff', type=int,
            help="cutoff on matchbox score, default=35")
    parser.add_argument('--no-show', 
            help="don't show final image",
            action="store_false")
    args = parser.parse_args()

    fsearch(im1_name = args.master,
        im2_name = args.child,
        method = args.method,
        show = args.show,
        matchbox = args.matchbox,
        score_cutoff = args.score-cutoff,
        angle_cutoff = args.angle-cutoff,
        show_final = args.no-show)
    

def fsearch(im1_name, im2_name, method="surf", show=False, matchbox=False, 
        score_cutoff=100, angle_cutoff=35, show_final=True):
    return_images = {}

    # skimage loads with color and selective search requires color
    img1 = skimage.io.imread(im1_name)
    img1 = imutils.resize(img1, width=500)

    img2 = skimage.io.imread(im2_name)
    img2 = imutils.resize(img2, width=500)

    master_img = img1
    search_img = img2

    # load in grayscale. 
    # skimage rgb2gray doesn't work with sift and surf
    #   - for further info see detect_organized_t.py 
    img1 = cv2.imread(im1_name,0)
    img1 = imutils.resize(img1, width=500)

    img2 = cv2.imread(im2_name,0)
    img2 = imutils.resize(img2, width=500)

    # s_search
    boxes = roi.s_search(search_img)
    s_search_img = display.just_boxes(boxes, search_img, show=show)
    return_images['s_search'] = s_search_img

    # checking rois with crosscheck 
    # IMPORTANT: surf works better than sift, 
    #            crosccheck should always be on
    #            crosscheck doesn't work with FLANN
    kp_master, rois = roi.check_roi_good(img1, img2, boxes, method, 
                                        modus="bf", crosscheck=True)

    # basic_cutoff
    idxs_basic = score.basic_cutoff(rois, 100)
    basic_cutoff_img = display.just_boxes_r(rois, search_img, idxs_basic, show=show)
    rois = trim.idx_trim(rois, idxs_basic)
    return_images['basic_cutoff'] = basic_cutoff_img
    

    # homography
    trim.homography_all(kp_master, master_img, search_img, rois)
    homography_img = display.homography_boxes(rois, search_img, show=show)
    return_images['homography'] = homography_img
    if matchbox:
        homography_matchboxes = display.matchbox(kp_master, master_img,
                search_img, rois, homography=True, show=show)
        return_images['homography_matchboxes'] = homography_matchboxes

    # homography angle cutoff
    idxs_angle = score.angle_cutoff(rois, angle_cutoff)
    rois = trim.idx_trim(rois, idxs_angle)
    angle_cutoff_img = display.homography_boxes(rois, search_img, show=show)
    return_images['angle_cutoff'] = angle_cutoff_img
    if matchbox:
        angle_cutoff_matchboxes = display.matchbox(kp_master, master_img,
                search_img, rois, homography=True, mMask=True, show=show)
        return_images['angle_cutoff_matchboxes'] = angle_cutoff_matchboxes

    # NMS
    idxs_nms = trim.nms_homography(rois, .50, score._mask_basic_s)
    rois = trim.idx_trim(rois, idxs_nms)
    nms_img = display.homography_boxes(rois, search_img,show=(show or show_final))
    return_images['nms'] = nms_img
    return_images['final'] = nms_img

    return rois, return_images