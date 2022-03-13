#!/usr/bin/env python3

import colorsys
import os
import pickle
import struct
import time

import cv2
import numpy as np
import sys
import glob
import yaml
import pprint


from squeezedet.squeezedet_classifier import SqueezeDetClassifier
from squeezedet.utils import util
from nanonets_object_tracking.deepsort import deepsort_rbc

possible_classes = ['S40_40_G', 'F20_20_B', 'BEARING', 'R20']

def convert_bboxes(bboxes):
    detections = []

    for i in range(len(bboxes)):

        bbox = util.bbox_transform(bboxes[i])
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        w = int(bbox[2] - bbox[0])
        h = int(bbox[3] - bbox[1])
        detections.append([x1, y1, w, h])


def visualize(frame, tracker, detections_class):
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        
        bbox = track.to_tlbr() #Get the corrected/predicted bounding box
        id_num = str(track.track_id) #Get the ID for the particular track.
        features = track.features #Get the feature vector corresponding to the detection.

        #Draw bbox from tracker.
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
        cv2.putText(frame, str(id_num),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

        #Draw bbox from detector. Just to compare.
        for det in detections_class:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,0), 2)
    
    cv2.imshow('frame',frame)
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return True
    return False

if __name__ == '__main__':
    config_file = 'squeezedet/rgb_classifier_config.yaml'

    if os.path.isfile(config_file):
        configs = {}
        with open(config_file, 'r') as infile:
            configs = yaml.safe_load(infile)

        model_config = configs['model']['squeezeDet']
        classes = configs['classes']
        colors = configs['colors']
        model_dir = 'squeezedet/model'

        model = SqueezeDetClassifier(config=model_config,
                                     checkpoint_path=model_dir)
        objects = []

        images = [(cv2.imread(file), file.split('/')[1]) for file in sorted(glob.glob("data/images/*.jpg"))]

        # deepsort
        deepsort = deepsort_rbc(wt_path='nanonets_object_tracking/ckpts/model640.pt')

        counter = 0
        for image, file in images:
            bboxes, probs, labels = model.classify(image)

            detections = convert_bboxes(bboxes)

            tracker, detections_class = deepsort.run_deep_sort(image, probs, detections)

            if_quit = visualize(image, tracker, detections_class)

            if if_quit:
                break
            # for i in range(len(labels)):
            #     result = {}

            #     result['name'] = classes[labels[i]]
            #     result['probability'] = probs[i]
            #     roi = {}
            #     bbox = util.bbox_transform(bboxes[i])
            #     roi['x_offset'] = int(bbox[0])
            #     roi['y_offset'] = int(bbox[1])
            #     roi['width'] = int(bbox[2] - bbox[0])
            #     roi['height'] = int(bbox[3] - bbox[1])
            #     result['roi'] = roi

            #     objects.append(result)
            #     cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 1)
            #     cv2.putText(image, classes[labels[i]], (int(bbox[0]), int(bbox[3]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #                 (255, 255, 255), 1, cv2.LINE_AA)

            # cv2.imwrite(os.path.join('annotated/', file), image)
            # counter += 1
            # #cv2.imshow('annotated', image)
            # cv2.waitKey(0)
            # print('\n', file)
            # pprint.pprint(objects)
