"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

import glob
import time
import argparse
from filterpy.kalman import KalmanFilter

np.random.seed(0)
global_counter = 0


def linear_assignment(cost_matrix):
    """
  Function to solve the association cost matrix
  """
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  
  I'm assuming this refers to the coordinates of the rectangle
  Are the inputs lists of bboxes or just individual ones?
  """
    global global_counter
    bb_gt = np.expand_dims(bb_gt, 0)  # Seems like this is converted to column vector
    bb_test = np.expand_dims(bb_test, 1)  # Converted to a row vector

    # We take the coordinates of the intersection rectangle
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)

    # Area of intersection
    wh = w * h  # element-wise multiplication?

    # IoU = intersection / (area_bb_test + area_bb_gt - intersection)
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    # I am assuming the shape would be no_of_detections x no_of_tracks
    global_counter += 1
    return (o)


def convert_bbox_to_z(bbox):
    """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))  # Make sure its in row format


# What is with the function naming, why not "convert_z_to_bbox"
def convert_x_to_bbox(x, score=None):
    """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
    w = np.sqrt(x[2] * x[3])  # w = np.sqrt(s*r)
    h = x[2] / w  # h = s/w,    cuz s is the area
    if (score == None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))
        # Here its a column vector


class KalmanBoxTracker(object):
    """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
    count = 0

    def __init__(self, bbox):
        """
    Initialises a tracker using initial bounding box.
    """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)  # State vector size = 7 and observation vector size = 4
        # state_vector_x = [u, v, s, r, u??, v??, ???]

        # F is the state transition matrix
        # u -> Only dependent upon previous u and u_dot
        # v -> Only dependent upon previous v and v_dot
        # s -> Only dependent upon previous s and s_dot
        # r -> Only depends on previous r (and I guess perturbations are only introduced by noise)
        # In a similar manner, u_dot, v_dot and s_dot are dependent only on their own previous states
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])

        # H is the measurement function
        # Only the first 4 elements [u, v, s, r] are observable
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        # Measurement uncertainty noise
        # Are we assuming the first 2 variables are almost perfectly measured? and that velocities
        # and scales are more prone to error
        self.kf.R[2:, 2:] *= 10.

        # Covariance matrix
        # I think this also says that the covariance is based only on the previous value of the same
        # state variable
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.

        # Process/State uncertainty/noise
        # I don't understand this operation, why are we reducing the uncertainty of these variables
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # Filter state estimate, seems like we set the initial state to the detection bbox
        self.kf.x[:4] = convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count  # Is this a global class variable independent of object?
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0  # assuming this means matches
        self.hit_streak = 0
        self.age = 0
        self.min_hits_satisfied = False     # Set to True of minimum number of detections have been associated


    def update(self, bbox):
        """
    Updates the state vector with observed bbox.

    It seems that this is used when we have found a suitable detection
    which can be associated with the previous frame bbox 
    """
        self.time_since_update = 0  # We reset the time since update
        self.history = []  # But why do we reset the history?
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))  # pass bbox in [x,y,s,r] format

    def predict(self):
        """
    Advances the state vector and returns the predicted bounding box estimate.
    """
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):  # Check if bbox is empty, i.e. no area
            self.kf.x[6] *= 0.0             # If yes, set it to 0
        self.kf.predict()                   # Predict the next state
        self.age += 1                       # Increase age
        if (self.time_since_update > 0):    # Reset streak if required
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]  # return the prediction

    def get_state(self):
        """
    Returns the current bounding box estimate.
    """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
    if (len(trackers) == 0):  # If no tracks exist, return the 1st and 3rd list as empty
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:  # i.e either detections or tracks should be present
        a = (iou_matrix > iou_threshold).astype(np.int32)  # We note all the instances where we have significant overlap
        # Ensure that all rows and columns add up to one, i.e. all tracks and
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            # We then return a list of indices which have been matched, each of these are tuples
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            # Else we assign these using a linear assignment method
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    # We then create the list of detections and tracks which weren't associated
    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # Filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            # With those that satisfy the criteria, we append them to a new list
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        # Convert the list to a numpy matrix
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
    Sets key parameters for SORT
    """
        # Max time before tracks without detections are terminated,
        self.max_age = max_age
        # Min hits to be considered a track? Probationary period
        self.min_hits = min_hits
        # Min threshold for comparison of detections and iou_thresholds
        self.iou_threshold = iou_threshold
        # Does this mean the IDs of the tracks?
        self.trackers = []
        self.labels = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5)), label=None):
        """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
        if label is None:
            label = []
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))  # Empty place holders equal to the number of tracks
        to_del = []
        ret = []
        ret_labels = []
        # We cycle through all the (active?) tracks
        for t, trk in enumerate(trks):
            # Predict the next position
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):  # remove such values, but not immediately
                to_del.append(t)
        # Find the nan values, mask them and them remove these
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        # So everything which has an incorrect prediction, we delete these
        for t in reversed(to_del):
            self.trackers.pop(t)  # Remove from active tracks?

        # We then associate the detections and tracks
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])  # we update the current position of the stored tracks

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            # print(i)
            trk = KalmanBoxTracker(dets[i])
            self.trackers.append(trk)
            self.labels.append(label[i])
        i = len(self.trackers)
        for label, trk in zip(reversed(self.labels),reversed(self.trackers)):
            d = trk.get_state()[0]
            if trk.hit_streak >= self.min_hits: # We just want the min_hits condition to be satisfied once
                trk.min_hits_satisfied = True
            if (trk.time_since_update < self.max_age) and (trk.min_hits_satisfied or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
                ret_labels.append(label)
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
                self.labels.pop(i)
        if(len(ret)>0):
            return np.concatenate(ret), ret_labels
        return np.empty((0,5)), []


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',
                        action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age",
                        help="Maximum number of frames to keep alive a track without associated detections.",
                        type=int, default=1)
    parser.add_argument("--min_hits",
                        help="Minimum number of associated detections before track is initialised.",
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # all train
    args = parse_args()
    display = args.display
    phase = args.phase
    total_time = 0.0
    total_frames = 0
    colours = np.random.rand(32, 3)  # used only for display
    if (display):
        if not os.path.exists('mot_benchmark'):
            print(
                '\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
            exit()
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(111, aspect='equal')

    if not os.path.exists('output'):
        os.makedirs('output')
    pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')
    # Cycle through the avaialble detections for each dataset
    for seq_dets_fn in glob.glob(pattern):
        # Initialize 1 tracker per dataset
        mot_tracker = Sort(max_age=args.max_age,
                           min_hits=args.min_hits,
                           iou_threshold=args.iou_threshold)  # Create instance of the SORT tracker
        seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')  # Is this a 2D numpy array?
        # Store the sequence name
        seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]

        with open(os.path.join('output', '%s.txt' % (seq)), 'w') as out_file:
            print("Processing %s." % (seq))
            for frame in range(int(seq_dets[:, 0].max())):
                frame += 1  # detection and frame numbers begin at 1
                dets = seq_dets[seq_dets[:, 0] == frame, 2:7]  # Interesting indexing
                dets[:, 2:4] += dets[:, 0:2]  # convert from [x1,y1,w,h] to [x1,y1,x2,y2]
                total_frames += 1

                if (display):
                    fn = os.path.join('mot_benchmark', phase, seq, 'img1', '%06d.jpg' % (frame))
                    im = io.imread(fn)
                    ax1.imshow(im)
                    plt.title(seq + ' Tracked Targets')

                # Actual tracking part
                start_time = time.time()
                trackers = mot_tracker.update(dets)  # seems like each is in [x1,y1,x2,y2] format
                cycle_time = time.time() - start_time
                total_time += cycle_time

                for d in trackers:
                    print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]),
                          file=out_file)
                    if (display):
                        d = d.astype(np.int32)
                        ax1.add_patch(patches.Rectangle((d[0], d[1]), d[2] - d[0], d[3] - d[1], fill=False, lw=3,
                                                        ec=colours[d[4] % 32, :]))

                if (display):
                    fig.canvas.flush_events()
                    plt.draw()
                    ax1.cla()

    print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (
    total_time, total_frames, total_frames / total_time))

    if (display):
        print("Note: to get real runtime results run without the option: --display")
