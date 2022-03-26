#!/usr/bin/env python3

############################################################
# Code for running Squeezedet with DeepSort. Just run 'python main.py' an enjoy :-).
############################################################

import colorsys
import os
import pickle
import struct
import time

import cv2
import numpy as np
import sys
import glob

import torch
import yaml
import pprint
import tensorflow 
import time

from torchvision.ops import nms

if tensorflow.__version__.startswith("2"):
    import tensorflow.compat.v1 as tf
else:
    import tensorflow as tf
tf.disable_v2_behavior()

sys.path.append('../')

from squeezedet.squeezedet_classifier import SqueezeDetClassifier
from squeezedet.utils import util
from sort import sort
from sort.sort import Sort
# from nanonets_object_tracking.deepsort import deepsort_rbc

possible_classes = ['S40_40_G', 'F20_20_B', 'BEARING', 'R20']

def convert_bboxes(bboxes, scores):
    detections = []

    for i in range(len(bboxes)):

        bbox = util.bbox_transform(bboxes[i])
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        w = int(bbox[2] - bbox[0])
        h = int(bbox[3] - bbox[1])
        detections.append([x1, y1, w, h, scores[i]])
    
    return detections

def nms_adapted(bboxes, scores):
    """
    Parameters:
        bboxes  [list[list]]
            Format of detections is `cwh` (x,y,w,h)
        scores  [list]

    Returns:
        detections  ndarray
            array of detections in `tlbr, scores` format
    """
    bboxes = torch.tensor(bboxes)

    # Convert from tlwh -> tlbr
    new_bboxes = torch.zeros(bboxes.size())
    new_bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2
    new_bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2
    new_bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2] / 2
    new_bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3] / 2

    scores = torch.tensor(scores)

    keep = nms(new_bboxes, scores, iou_threshold=0.1)

    new_bboxes = new_bboxes[keep].tolist()
    scores = scores[keep].tolist()

    detections = [[new_bboxes[i][0], new_bboxes[i][1], new_bboxes[i][2], new_bboxes[i][2], scores[i]] for i in range(len(new_bboxes))]
    return np.array(detections)

# def visualize(frame, tracker, detections_class):
#     for track in tracker.tracks:
#         if not track.is_confirmed() or track.time_since_update > 1:
#             continue
        
#         bbox = track.to_tlbr() #Get the corrected/predicted bounding box
#         id_num = str(track.track_id) #Get the ID for the particular track.
#         features = track.features #Get the feature vector corresponding to the detection.

#         #Draw bbox from tracker.
#         cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
#         cv2.putText(frame, str(id_num),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

#         #Draw bbox from detector. Just to compare.
#         for det in detections_class:
#             bbox = det.to_tlbr()
#             cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,0), 2)
    
#     cv2.imshow('frame',frame)
   
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         return True
#     return False

def visualize(frame, tracks):
    for track in tracks:
        bbox = track[0:4]
        id_num = track[4]

        #Draw bbox from tracker.
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
        cv2.putText(frame, str(id_num),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

        #Draw bbox from detector. Just to compare.
        # for det in detections_class:
        #     bbox = det.to_tlbr()
        #     cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,0), 2)
    
    cv2.imshow('frame',frame)
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return True
    return False


def visualize_dets(frame, bboxes):
    for bbox in bboxes:
        # bbox = track[0:4]
        # id_num = track[4]

        # bbox[2:4] += bbox[0:2]
        # Draw bbox from tracker.
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
        # cv2.putText(frame, str(id_num), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)

        # Draw bbox from detector. Just to compare.
        # for det in detections_class:
        #     bbox = det.to_tlbr()
        #     cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        return True
    return False



if __name__ == '__main__':
    config_file = '../squeezedet/rgb_classifier_config.yaml'

    if os.path.isfile(config_file):
        configs = {}
        with open(config_file, 'r') as infile:
            configs = yaml.safe_load(infile)

        model_config = configs['model']['squeezeDet']
        classes = configs['classes']
        colors = configs['colors']
        model_dir = '../squeezedet/model'

        model = SqueezeDetClassifier(config=model_config,
                                     checkpoint_path=model_dir)
        objects = []

        images = [(cv2.imread(file), file.split('/')[1]) for file in sorted(glob.glob("../data/images/*.jpg"))]

        # deepsort
        # deepsort = deepsort_rbc(wt_path='../nanonets_object_tracking/ckpts/model640.pt')

        detector_rate = 5
        # SORT
        sort_tracker = Sort(max_age=3, 
                            min_hits=2,
                            iou_threshold=0.1)

        counter = 0
        i = 0
        for i, (image, file) in enumerate(images):
            print(f"Frame number {i}")
            if i%detector_rate==0 or i%detector_rate==1 or i%detector_rate==2:
                # Returns bboxes in cwh format
                bboxes, probs, labels = model.classify(image)
                # visualize_dets(image, np.array(bboxes))

                # Combines scores and bboxes, and converts to tlbr format
                detections = nms_adapted(bboxes, probs)

                visualize_dets(image, detections[:, :4])

                if detections is None:
                    print("No dets")
                    continue

                trackers = sort_tracker.update(detections)  # This returns bbox and track_id
            else:
                trackers = sort_tracker.update()
            i = i+1
            
            # print(type(trackers), trackers.shape, trackers)
            # if_quit = visualize(image, trackers)
            # if if_quit:
            #     break
