import detect_organized as det
import roi
import display
import score
import trim

import cv2
import imutils
import skimage
import argparse
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
    parser.add_argument('-bc','--basic-cutoff', type=int,
            help="cutoff on matchbox basic score, default=15",
            const=15)
    parser.add_argument('-ac','--angle-cutoff', type=int,
            help="cutoff on matchbox angle in degrees, default=35",
            const=35)
    parser.add_argument('-ot','--overlap-thresh', type=float,
            help="threshold on overlap, default=.3",
            const=.3)
    parser.add_argument('--no-show', 
            help="don't show final image",
            action="store_false")
    parser.add_argument('-sm','--search-mode', 
            help="selective search mode: \"single\", \"fast\", \"quality\"",
            const="fast")
    args = parser.parse_args()

    fsearch(im1_name = args.master,
        im2_name = args.child,
        method = args.method,
        show = args.show,
        matchbox = args.matchbox,
        basic_score_cutoff = args.basic_score_cutoff,
        angle_cutoff = args.angle_cutoff,
        overlap_thresh = args.overlap_thresh,
        show_final = args.no_show,
        search_mode = args.search_mode)
    

def fsearch(im1_name, im2_name, method="surf", show=False, matchbox=False, 
        basic_score_cutoff=15, angle_cutoff=35, overlap_thresh=.3, show_final=True, search_mode='fast'):
    return_images = {}

    kp_master, rois = roi_search (im1_name, im2_name, return_images, method=method,
                                show=show, search_mode=search_mode)

    rois = prune (kp_master, rois, return_images, show=show, matchbox=matchbox, 
        basic_score_cutoff=basic_score_cutoff, angle_cutoff=angle_cutoff, 
        overlap_thresh=overlap_thresh, show_final=show_final)

    return rois, return_images

def roi_search(im1_name, im2_name, return_images, method="surf", show=False, search_mode='fast'):
    # skimage loads with color and selective search requires color
    img1 = skimage.io.imread(im1_name)
    img1 = imutils.resize(img1, width=500)

    img2 = skimage.io.imread(im2_name)
    img2 = imutils.resize(img2, width=500)

    master_img = img1
    search_img = img2
    return_images['master'] = master_img
    return_images['search'] = search_img

    # load in grayscale. 
    # skimage rgb2gray doesn't work with sift and surf
    #   - for further info see detect_organized_t.py 
    img1 = cv2.imread(im1_name,0)
    img1 = imutils.resize(img1, width=500)

    img2 = cv2.imread(im2_name,0)
    img2 = imutils.resize(img2, width=500)

    # s_search
    boxes = roi.s_search(search_img, search_mode)
    if show: print('s search')
    s_search_img = display.just_boxes(boxes, search_img, show=show)
    return_images['s_search'] = s_search_img

    # checking rois with crosscheck 
    # IMPORTANT: surf works better than sift, 
    #            crosccheck should always be on
    #            crosscheck doesn't work with FLANN
    kp_master, rois = roi.check_roi_good(img1, img2, boxes, method, 
                                        modus="bf", crosscheck=True)

    return kp_master, rois

def prune(kp_master, rois, return_images, show=False, matchbox=False, basic_score_cutoff=15,
     angle_cutoff=35, overlap_thresh=.3, show_final=True, score_fn=score._mask_basic_s):
    master_img = return_images['master']
    search_img = return_images['search']
    # basic_cutoff
    idxs_basic = score.basic_cutoff(rois, basic_score_cutoff)
    if show: print('basic_score_cutoff')
    score_cutoff_img = display.just_boxes_r(rois, search_img, idxs_basic, show=show)
    rois = trim.idx_trim(rois, idxs_basic)
    return_images['basic_score_cutoff'] = score_cutoff_img
    

    # homography
    trim.homography_all(kp_master, master_img, search_img, rois)
    if show: print('homography')
    homography_img = display.homography_boxes(rois, search_img, show=show)
    return_images['homography'] = homography_img
    if matchbox:
        homography_matchboxes = display.matchbox(kp_master, master_img,
                search_img, rois, homography=True, show=False)
        return_images['homography_matchboxes'] = homography_matchboxes

    # homography angle cutoff
    idxs_angle, angles = score.angle_cutoff(rois, angle_cutoff)
    rois = trim.idx_trim(rois, idxs_angle)
    assert len(angles) == len(rois)
    for i, roii in enumerate(rois):
        roii[5] = angles[i]
    if show: print('angle cutoff')
    angle_cutoff_img = display.homography_boxes(rois, search_img, show=show)
    return_images['angle_cutoff'] = angle_cutoff_img
    if matchbox:
        angle_cutoff_matchboxes = display.matchbox(kp_master, master_img,
                search_img, rois, homography=True, mMask=True, show=False)
        return_images['angle_cutoff_matchboxes'] = angle_cutoff_matchboxes

    # NMS
    idxs_nms = trim.nms_homography(rois, overlap_thresh, score_fn)
    rois = trim.idx_trim(rois, idxs_nms)
    if show or show_final: print('final')
    nms_img = display.homography_boxes(rois, search_img,show=(show or show_final))
    return_images['nms'] = nms_img
    return_images['final'] = nms_img

    return rois