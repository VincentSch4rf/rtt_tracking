#!/usr/bin/env python3

############################################################
# Code for running Squeezedet with DeepSort. Just run 'python main.py' an enjoy :-).
############################################################

import colorsys
import os
import pickle
import struct
import time
from pathlib import Path

import cv2
import numpy as np
import sys
import glob

import torch
import yaml
import pprint
import tensorflow
import time
import json
import argparse

from torchvision.ops import nms

import yolov5
from yolov5 import YOLOv5

from nanonets_object_tracking.yolo_detector import YoloDetector

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


def nms_adapted(bboxes, scores, labels):
    """
    Parameters:
        bboxes  [list[list]]
            Format of detections is `cwh` (x,y,w,h)
        scores  [list]
        labels  [list]
    Returns:
        detections  ndarray
            array of detections in `tlbr, scores` format
        labels  [list]
    """
    # Convert inputs to tensors
    bboxes = torch.tensor(bboxes)
    scores = torch.tensor(scores)
    labels = torch.tensor(labels)
    
    # Make sure bboxes is 2D
    if len(bboxes.size()) < 2:
        bboxes = bboxes.view(1, bboxes.size()[0])

    # Convert from cwh -> tlbr
    new_bboxes = torch.zeros(bboxes.size())
    new_bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2
    new_bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2
    new_bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2] / 2
    new_bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3] / 2

    # Perform NMS
    keep = nms(new_bboxes, scores, iou_threshold=0.1)

    # Convert tensors back to lists
    new_bboxes = new_bboxes[keep].tolist()
    scores = scores[keep].tolist()
    labels = labels[keep].tolist()

    # Remove bboxes masked by NMS and concatenate bboxes and scores to a single list
    detections = [[new_bboxes[i][0], new_bboxes[i][1], new_bboxes[i][2], new_bboxes[i][3],
                   scores[i]] for i in range(len(new_bboxes))]
    return np.array(detections), labels


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

def visualize(frame, tracks, obj_classes):
    """ Function to visualize the bounding boxes returned from the tracker
    Inputs:
        frame       Image ndarray
            The frame on which the bboxes are to be visualized
        tracks      nd array
            Each row has bbox coordinates and track id stored
            Bounding boxes are in 'tlbr' format
        obj_classes list
    """
    for track in tracks:
        bbox = track[0:4]
        id_num = track[4]

        # Draw bbox from tracker. bbox format is 'tlbr'
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
        cv2.putText(frame, str(id_num), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)

        # Draw bbox from detector. Just to compare.
        # for det in detections_class:
        #     bbox = det.to_tlbr()
        #     cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        return True
    return False


def store_results(annots, frame_id, tracks):
    """
    A failed vision for a beautiful function
    RIP :( You won't be forgotten
    Inputs:
        annots      Dict[Dict[list]]
                    Stores the output
        frame_id    int
        tracks      nd array
            Each row has bbox coordinates and track id stored
            Bounding boxes are in 'tlbr' format
    Returns:
        annots      Dict[Dict[list]]
                    Now updated with new frames
    """
    pass


def visualize_dets(frame, bboxes):
    """ Function to visualize the detection bounding boxes on the image
    Inputs:
        frame       image ndarray
            The frame on which the bboxes are to be visualized
        bboxes      list[nparray]
            Bounding boxes in 'tlbr' format
    """
    for bbox in bboxes:
        # Draw bboxes from the detector
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
        # cv2.putText(frame, str(id_num), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return True
    return False


def check_and_create_path(dir_path):
    """ Checks if a particular directory exists, if not, the directory is created
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="SqueezeDet SORT")
    parser.add_argument(
        "--max_age", help="Set the maximum age/frames for which a track "
        "can exist without being associated with detections.",
        default=11, type=int)
    parser.add_argument(
        "--min_hits", help="Set the minimum number of consecutive detections "
        "to be done in order for these set of detections to be considered a track.",
        default=3, type=int)
    parser.add_argument(
        "--iou_threshold", help="Minimum overlap between detection and estimation bboxes "
        "to be considered the same track.", type=float, default=0.1)
    parser.add_argument(
        "--low_frame_rate_modulo", help="Set the number of frames to be skipped. "
        "This is used to simulate low frame rates, by skipping n frames",
        default=1, type=int)
    parser.add_argument(
        "--display", help="Show intermediate tracking results ",
        default='True', type=str)
    parser.add_argument(    # Currently not used
        "--nms", help="Parameter to set if NMS is to be performed on the bounding boxes "
        "returned by the detector. This performs inter-class NMS Can be set to \'True\'"
        "or \'False\''. By default it is set to \'True\'", default="True", type=str)
    parser.add_argument(
        "--path_to_dataset", help="Set path to dataset "
        "default: ../data/images", default="../data/images", type=str)
    parser.add_argument(    # Currently not used
        "--dataset_split", help="Set the dataset split to run the tracker "
        "default: train. Can also use \'test\'", default="train", type=str)
    parser.add_argument(
        "--generate_outputs", help="Select whether outputs are to be generated"
        "or not default: True. Can also be set to \'False\'", default="True", type=str)
    parser.add_argument(
        "--path_to_annotations", help="Set the path to the output folder"
        "default: outputs", default='outputs', type=str)
    parser.add_argument(
        "--model_path", help="Path to [YOLOv5, SqueezeNet] model", default="models/rtt_376_1280.pt", type=str
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    config_file = '../squeezedet/rgb_classifier_config.yaml'

    if os.path.isfile(config_file):
        configs = {}
        with open(config_file, 'r') as infile:
            configs = yaml.safe_load(infile)

        model_config = configs['model']['squeezeDet']
        classes = configs['classes']
        colors = configs['colors']
        p = Path(args.model_path)
        if ".pt" in p.name:
            model = YoloDetector(checkpoint_path=args.model_path)
        else:
            model = SqueezeDetClassifier(config=model_config,
                                         checkpoint_path=args.model_path)
        objects = []

        images = [(cv2.imread(file), int(file.split('/')[-1].lstrip('frame').split('.')[0])) for file in sorted(glob.glob("/".join([args.path_to_dataset, "*.jpg"])))]
        # .split('/')[1]
        detector_rate = 10
        # SORT
        sort_tracker = Sort(args.max_age,
                            args.min_hits,
                            args.iou_threshold)

        # Add path and create a dict to store the output annotations
        check_and_create_path(args.path_to_annotations)
        annotations = {}

        j = 0
        for i, (image, frame_id) in enumerate(images):

            if i % args.low_frame_rate_modulo != 0:
                continue

            #if j % detector_rate == 0 or j % detector_rate == 1 or j % detector_rate == 2:
                # Returns bboxes in cwh format
            print(f"Frame number, global {i}")
            bboxes, scores, labels = model.classify(image) # bboxes format is tlbr
            detections = np.hstack((bboxes, np.atleast_2d(scores)))
            trackers, obj_classes = sort_tracker.update(detections, labels)  # This returns bbox and track_id

            if args.display == 'True':
                if_quit = visualize(image, trackers, obj_classes)
                if if_quit:
                    break
            j += 1

            # We store the tracking results
            if args.generate_outputs == 'True':
                annotations[frame_id] = {}
                for track in trackers:
                    annotations[frame_id][int(track[4])] = track[0:4].tolist()
                    # bbox format is 'tlbr'

        if args.generate_outputs == 'True':
            # We finally write the outputs to a .json file
            with open("/".join([args.path_to_annotations, 'sort_outputs.json']), "w") as fp:
                    json.dump(annotations,fp)
