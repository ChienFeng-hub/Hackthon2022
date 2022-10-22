from lib2to3.pgen2.tokenize import StopTokenizing
from platform import system
from re import S
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from motrackers import CentroidTracker, IOUTracker
from motrackers.utils import draw_tracks
import sys
import argparse

import platform
import shutil
import time
from pathlib import Path
import easyocr
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from car_detector.hybridnets import HybridNets, optimized_model

def write_bbox(img, bbox, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img

def read_box(bbox_path, frame_id):
    res_path = os.path.join(bbox_path, str(frame_id) + '.txt')
    with open(res_path, 'r') as f:
        lines = f.readlines()
        bbox = []
        for line in lines:
            bbox.append([int(float(x)) for x in line.split(',')[:4]])
    return bbox
def draw_all_box(img, bbox):
    for box in bbox:
        img = write_bbox(img, box)
    return img
def draw_text(img, text, pos, color=(0, 255, 0), thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, pos, font, 3, color, thickness, cv2.LINE_AA)
    return img
def letterbox(img, new_shape=(512, 512), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / \
            shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


# def tracking(img_dir, bbox_dir, res_path):

#     if not os.path.exists(res_path):
#         os.makedirs(res_path)
    
#     tracker = IOUTracker()
#     frame_len = len(os.listdir(img_dir))
    
#     global tracking_dict
#     tracking_dict = {}
#     global bboxes
#     bboxes = []
#     for i in range(frame_len):
#         img_path = os.path.join(img_dir, str(i) + '.jpg')
#         img = cv2.imread(img_path)
#         bbox = read_box(bbox_dir, i)
#         bboxes.append(bbox)
#         bbox = np.array(bbox)
#         detection_confidences = np.ones(bbox.shape[0])
#         detection_class_ids = np.zeros(bbox.shape[0])
#         output_tracks = tracker.update(bbox, detection_confidences, detection_class_ids)
#         img = draw_all_box(img, bbox)

#         for track in output_tracks:
#             frame, id, bb_left, bb_top, bb_width, bb_height, confidence, x, y, z = track
#             if id not in tracking_dict:
#                 tracking_dict[id] = [[frame-1, int(bb_left), int(bb_top), int(bb_width), int(bb_height), confidence, x, y, z]]
#             else:
#                 tracking_dict[id].append([frame-1, int(bb_left), int(bb_top), int(bb_width), int(bb_height), confidence, x, y, z])
#             img = draw_text(img, str(id), (int(bb_left), int(bb_top) - 5))
#             # print(img.shape)
#             # print(track)
#         # print("\n")
        
#         cv2.imwrite( os.path.join(res_path , str(i) + '.jpg'), img)
#     tmp = tracking_dict.copy()
#     for id in tmp.keys():
#         if len(tracking_dict[id]) < 60:
#             del tracking_dict[id]
#     del tmp
    
#     stop_list, stop_frame = Stop_detection(res_path)
#     ALPR(stop_list, img_dir, res_path, stop_frame)
    # return tracking_dict

def Stop_detection(res_path):

    window_size = 30
    area_threshold = 41000
    area_ratio_threshold = 0.999
    detect_id = None
    detect_stop = {}
    for id in tracking_dict.keys():
        for i in range(len(tracking_dict[id]) - window_size):
            bigger  = 0
            biggest_area = 0
            for j in range(window_size):
                area_before = (tracking_dict[id][i+j][3]-tracking_dict[id][i+j][1]) * (tracking_dict[id][i+j][4]-tracking_dict[id][i+j][2])
                area_after = (tracking_dict[id][i+j+1][3] - tracking_dict[id][i+j+1][1]) * (tracking_dict[id][i+j+1][4]-tracking_dict[id][i+j+1][2])
                # print(area_before, area_after)
                
                biggest_area = max(biggest_area, area_before, area_after)
                if (area_before / area_after < area_ratio_threshold):             
                    bigger += 1
                if bigger >= window_size//2:
                    detect_stop[id] = max(0, tracking_dict[id][i][0]-window_size//2)
        if(biggest_area > area_threshold) and (tracking_dict[id][i][1] + tracking_dict[id][i][3])//2 > 550 and id in detect_stop:
            print(id, 'is stopped at', detect_stop[id])
            detect_id = id
            for j in range(detect_stop[id], min(i+window_size, len(tracking_dict[id]))):
                stop_img = cv2.imread(os.path.join(res_path , str(j) + '.jpg'))
                stop_img = draw_text(stop_img, 'STOOOOOOOOP', (1000, 50))
                cv2.imwrite( os.path.join(res_path , str(j) + '.jpg'), stop_img)
    # print(detect_id)
    return tracking_dict[detect_id], detect_stop[detect_id]

def Redlight_detection(res_path, red_txt):
    red_light = []
    with open(red_txt, 'r') as f:
        for line in f:
            red_light.append(int(line.split(":")[1].replace(" ", "")))
    # print(red_light)
    window_size = 30
    area_threshold = 6000
    area_ratio_threshold = 1.001
    detect_id = None
    detect_stop = {}
    for id in tracking_dict.keys():
        for i in range(len(tracking_dict[id]) - window_size):
            smaller  = 0
            smallest_area = float('inf')
            
            for j in range(window_size):
                area_before = (tracking_dict[id][i+j][3]-tracking_dict[id][i+j][1]) * (tracking_dict[id][i+j][4]-tracking_dict[id][i+j][2])
                area_after = (tracking_dict[id][i+j+1][3] - tracking_dict[id][i+j+1][1]) * (tracking_dict[id][i+j+1][4]-tracking_dict[id][i+j+1][2])
                # print(area_before, area_after)
                
                smallest_area = min(smallest_area, area_before, area_after)
                if (area_before / area_after > area_ratio_threshold):             
                    smaller += 1
                if smaller >= window_size//2:
                    detect_stop[id] = max(0, tracking_dict[id][i][0]-smaller)
        # print(id, '_smallest_area', smallest_area)
        if(smallest_area>3000 and smallest_area < area_threshold and id in detect_stop):
            print(id, 'breaks redlight at', detect_stop[id])
            # print(smallest_area)
            detect_id = id
            for j in range(detect_stop[id], min(i+window_size, len(tracking_dict[id]))):
                stop_img = cv2.imread(os.path.join(res_path , str(j) + '.jpg'))
                stop_img = draw_text(stop_img, 'Redlight', (1300, 100), color=(0, 0, 0), thickness=5)
                cv2.imwrite( os.path.join(res_path , str(j) + '.jpg'), stop_img)
    # print(detect_id)
    # print(detect_stop.keys())
    return tracking_dict[detect_id], detect_stop[detect_id]


def ALPR(detect_list, img_dir, res_path, stop_frame):
    reader = easyocr.Reader(['en'])
    max_conf = 0
    plate_num = "not detected"
    for info in detect_list:
        img = cv2.imread(os.path.join(img_dir, str(info[0])+".jpg"))
        img0 = img[info[2]:info[4], info[1]:info[3]]
        # cv2.imwrite(os.path.join("fuck", str(info[0])+".jpg"), img0)
        results = reader.readtext(img0)

        for res in results:
            # print(res)
            if(res[2]>max_conf and len(res[1])>=5 and len(res[1])<=8):
                plate_num = res[1]
                max_conf = res[2]
    plate_num = plate_num.replace(" ", "")
    print(plate_num)
    if not plate_num == "not detected":
        for info in detect_list:
            if info[0] >= stop_frame:
                img_text = cv2.imread(os.path.join(res_path, str(info[0])+".jpg"))
                img_text = draw_text(img_text, plate_num, (50, 100), color=(0, 0, 0), thickness=5) 
                # print(plate_num)
                cv2.imwrite(os.path.join(res_path, str(info[0])+".jpg"), img_text)

        # end = time.time()
        # print('Time::', end-start)

def detection(video_path, image_dir, plt_dir, red_txt=None):
    model_path = "car_detector/models/hybridnets_384x512/hybridnets_384x512.onnx"
    anchor_path = "car_detector/models/hybridnets_384x512/anchors_384x512.npy"
    optimized_model(model_path) # Remove unused nodes
    car_detector = HybridNets(model_path, anchor_path, conf_thres=0.5, iou_thres=0.5)
    tracker = IOUTracker(max_lost=2, iou_threshold=0.5, min_detection_confidence=0.4, max_detection_confidence=0.7,
                         tracker_output_format='mot_challenge')

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(plt_dir):
        os.makedirs(plt_dir)
    cap = cv2.VideoCapture(video_path)
    global tracking_dict
    tracking_dict = {}
    global bboxes
    bboxes = []

    i = 0
    while True:
        ok, image = cap.read()
        
        if not ok:
            # print("Cannot read the video feed.")
            break
        image = image[:950, :, :]
        seg_map, filtered_boxes, _ = car_detector(image)
        bboxes.append(filtered_boxes)
        filtered_boxes = np.array(filtered_boxes)
        filtered_scores = np.ones(filtered_boxes.shape[0])
        class_ids = np.ones_like(filtered_scores).astype(int)
        
        output_tracks = tracker.update(filtered_boxes, filtered_scores, class_ids=class_ids)

        updated_image = car_detector.draw_2D(image)
        # updated_image = draw_tracks(updated_image, output_tracks)
        for track in output_tracks:
            frame, id, bb_left, bb_top, bb_width, bb_height, confidence, x, y, z = track
            if id not in tracking_dict:
                tracking_dict[id] = [[frame-1, int(bb_left), int(bb_top), int(bb_width), int(bb_height), confidence, x, y, z]]
            else:
                tracking_dict[id].append([frame-1, int(bb_left), int(bb_top), int(bb_width), int(bb_height), confidence, x, y, z])
            updated_image = draw_text(updated_image, str(id), (int(bb_left), int(bb_top) - 20))
        cv2.imwrite(os.path.join(image_dir, str(i) + '.jpg'), image)
        cv2.imwrite(os.path.join(plt_dir, str(i) + '.jpg'), updated_image)
        i += 1
    cap.release()

    tmp = tracking_dict.copy()
    for id in tmp.keys():
        if len(tracking_dict[id]) < 60:
            del tracking_dict[id]
    del tmp

    if red_txt != None:
        red_list, red_frame = Redlight_detection(plt_dir, red_txt)
        ALPR(red_list, image_dir, plt_dir, red_frame)
    else:
        stop_list, stop_frame = Stop_detection(plt_dir)
        ALPR(stop_list, image_dir, plt_dir, stop_frame)
# detect()
# tracking('./redlight/1', './redlight_bbox/1')
# tracking('./stop/1', './stop_bbox/1', './result_stop')

# detection('./stop.mp4', './stop_frames', './stop_plt')

detection('./r.mp4', './redlight_frames', './redlight_plt', './redlight1.txt')



        