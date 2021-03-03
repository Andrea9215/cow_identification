'''
Main script for the cow identification using yolo_net and EasyOCR

Steps:
- Initialization
    load image list
    yolo_net
    easyocr

- Apply yolo_net
    Detect the cow's face pose and determine if its drinking or not.
    Detect the position of the tag if its visible

- Apply some filters on tag bbox (not so usefull)

- Apply EasyOCR
    if detected on the tag's bbox
    else assume that tag is not visible

TO DO
- Check the detected tag number with the tag list
'''

from yolo import yoloManager
import cv2
import glob
import os
import time
import numpy as np
import easyocr
import matplotlib.pyplot as plt

def put_ocr_detection(image, result):
    for bbox, text, prob in result:
        p1 = (int(bbox[0][0]),int(bbox[0][1]))
        p2 = (int(bbox[1][0]),int(bbox[1][1]))
        p3 = (int(bbox[2][0]),int(bbox[2][1]))
        p4 = (int(bbox[3][0]),int(bbox[3][1]))

        image = cv2.line(image, p1, p2, (0,255,255),2)
        image = cv2.line(image, p2, p3, (0,255,255),2)
        image = cv2.line(image, p3, p4, (0,255,255),2)
        image = cv2.line(image, p4, p1, (0,255,255),2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        color = (255,255,0)
        thickness = 2

        image = cv2.putText(image, text, p4, font, font_scale, color, thickness, cv2.LINE_AA, False)

# Load the net to detect the face and the tag o cow
id_net = yoloManager.YoloManager()
id_net.loadNet()

reader = easyocr.Reader(['en'], gpu=True) # need to run only once to load model into memory

# Load images list
path_images = os.path.join('images_output', '*.jpg')
list_images = glob.glob(path_images)

# loop on images
for image_name in list_images:
    # load images
    image = cv2.imread(image_name)
    cv2.imshow('Input image', image)

    # apply yolo net
    outputs = id_net.applyNet(image)

    for class_id, bbox, confidence, time_stamp in outputs:
        # if tag detected look for tag edges
        if class_id == 1:
            x, y, w, h = bbox
            x_min = int(x-w/2)
            x_max = int(x+w/2)
            y_min = int(y-h/2)
            y_max = int(y+h/2)
            tag = image[y-h:y+h, x-w:x+w]
            cv2.imshow('Tag', tag)
        else:
            continue

        # grayscale
        tag_gray = cv2.cvtColor(tag, cv2.COLOR_BGR2GRAY)
        tag_gray = cv2.GaussianBlur(tag_gray, (11,11),0)
        cv2.imshow('Tag_gray', tag_gray)

        # Get structural element
        blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 25))
        tag_blackhat = cv2.morphologyEx(tag_gray, cv2.MORPH_BLACKHAT, blackhat_kernel)
        cv2.imshow('Blackhat', tag_blackhat)

################################################################################
        # # Binary
        # ret,tag_thresh = cv2.threshold(tag_blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # # tag_thresh = cv2.adaptiveThreshold(tag_blackhat,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,71,2)
        # # tag_thresh = cv2.GaussianBlur(tag_thresh, (11,11),0)
        # # tag_thresh = cv2.cvtColor(tag_thresh, cv2.COLOR_GRAY2BGR)
        # cv2.imshow('Tag_thresh__', tag_thresh)
        #
        # # Erode bynary
        # kernel = np.ones((3,3),np.uint8)
        # erosion = cv2.erode(tag_thresh,kernel,iterations = 2)
        # cv2.imshow('Erosion', erosion)

################################################################################
        # Apply OCR
        result = reader.readtext(tag_blackhat)
        print(result)

        # DISPLY INFORMATION
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        color = (255,255,0)
        thickness = 2

        if result:
            put_ocr_detection(tag, result)
            for bbox, text, prob in result:
                tag = cv2.putText(tag, text, (10,20), font, font_scale, color, thickness, cv2.LINE_AA, False)
                # put_info(image, text, yolo_time=time_yolo, ocr_time=time_ocr, detected_tag=detected_tag)

        cv2.imshow('ID', tag)

        # rho = 1  # distance resolution in pixels of the Hough grid
        # theta = np.pi / 180  # angular resolution in radians of the Hough grid
        # threshold = 20  # minimum number of votes (intersections in Hough grid cell)
        # min_line_length = 35  # minimum number of pixels making up a line
        # max_line_gap = 6  # maximum gap in pixels between connectable line segments
        # line_image = np.copy(tag) * 0  # creating a blank to draw lines on
        #
        # # Run Hough on edge detected image
        # # Output "lines" is an array containing endpoints of detected line segments
        # lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
        #                     min_line_length, max_line_gap)
        # if  lines is None:
        #     continue
        # points = []
        # for line in lines:
        #     for x1, y1, x2, y2 in line:
        #         points.append(((x1 + 0.0, y1 + 0.0), (x2 + 0.0, y2 + 0.0)))
        #         tag = cv2.line(tag, (x1, y1), (x2, y2), (255, 0, 0), 3)
        #
        # cv2.imshow('tag_line', tag)
        # rotate tag



    key = cv2.waitKey(0)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
