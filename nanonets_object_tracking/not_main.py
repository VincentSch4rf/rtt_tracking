#!/usr/bin/env python3

############################################################
# Code for running Squeezedet with DeepSort. Just run 'python main.py' an enjoy :-).
############################################################

import os
import sys

sys.path.append('../')

import cv2
import sys
import glob
import yaml
import rospy
from sensor_msgs.msg import Image
import torch
from torchvision.ops import nms

from mas_perception_msgs.msg import ImageList
from cv_bridge import CvBridge
from squeezedet.squeezedet_classifier import SqueezeDetClassifier
from squeezedet.utils import util
from nanonets_object_tracking.deepsort import deepsort_rbc
from nanonets_object_tracking.sort.sort import Sort
import time
import numpy as np

CLASS_NAMES =('s40_40_G', 's40_40_B', 'r20','motor',
                            'm30', 'm20_100', 'm20', 'f20_20_G', 'f20_20_B',
                            'em_02', 'em_01', 'distance_tube', 'container_box_red',
                            'container_box_blue', 'bearing_box_ax16',
                            'bearing_box_ax01', 'bearing', 'axis')

def convert_bboxes(bboxes):
    detections = []

    for i in range(len(bboxes)):

        bbox = util.bbox_transform(bboxes[i])
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        w = int(bbox[2] - bbox[0])
        h = int(bbox[3] - bbox[1])
        detections.append([x1, y1, w, h])
    
    return detections


def publish_annotations(frame, tracks, obj_classes):
    for track, obj_class in zip(tracks, obj_classes):
        bbox = track[0:4]
        id_num = track[4]

        #Draw bbox from tracker.
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
        cv2.putText(frame, str(id_num),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)
        cv2.putText(frame, str(CLASS_NAMES[obj_class]),(int(bbox[2]), int(bbox[3])),0, 5e-3 * 200, (0,255,0),2)

    image = bridge.cv2_to_imgmsg(frame, encoding="passthrough")
    publisher.publish(image)
   
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     return True
    # return False


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
    detections = [[new_bboxes[i][0], new_bboxes[i][1], new_bboxes[i][2], new_bboxes[i][3], \
                   scores[i]] for i in range(len(new_bboxes))]
    return np.array(detections), labels


def callback(data):
    #image = bridge.imgmsg_to_cv2(data.images[0], "bgr8") 
    image = bridge.imgmsg_to_cv2(data, "bgr8")

    global i, j, sort_tracker
    if j % detector_rate == 0 or j % detector_rate == 1 or j % detector_rate == 2:
        now = time.time()
        bboxes, probs, labels = model.classify(image)
        detection_time = time.time() - now
        # print('Detection', detection_time)
        # detections = convert_bboxes(bboxes)
        # Combines scores and bboxes, and converts to tlbr format
        detections, labels = nms_adapted(bboxes, probs, labels)

        if detections is None or len(detections.shape) == 0:
            print("No dets")
            return
        
        now = time.time()
        trackers, obj_classes = sort_tracker.update(detections, labels)  # This returns bbox and track_id
        print("with dets, No. of tracks:", len(trackers))
    else:
        now = time.time()
        trackers, obj_classes = sort_tracker.update()
        
        print("No det, No. of tracks:", len(trackers))
        detection_time = time.time() - now
        # print('Tracking', detection_time)

    publish_annotations(image, trackers, obj_classes)
    j += 1


def listener():
    global publisher
    rospy.init_node('listener', anonymous=True)

    #rospy.Subscriber("/mir_perception/multimodal_object_recognition/recognizer/rgb/input/images", ImageList, callback)
    rospy.Subscriber("/arm_cam3d/rgb/image_raw", Image, callback)
    publisher = rospy.Publisher('/image', Image, queue_size=1)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


config_file = '../squeezedet/rgb_classifier_config.yaml'
bridge = CvBridge()
publisher = None
    
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

    # images = [(cv2.imread(file), file.split('/')[1]) for file in sorted(glob.glob("../data/images/*.jpg"))]

    detector_rate = 10
    # SORT
    sort_tracker = Sort(max_age=11,
                        min_hits=3,
                        iou_threshold=0.1)
    i, j = 0, 0

    listener()
else:
    print('Could not find the file')
