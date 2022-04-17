from matplotlib.pyplot import show
import motmetrics as mm
import numpy as np
import json
import glob
import os
import argparse


def pretty(d, indent=0):
    """ Function to print dictionaries

        Inputs:                 [Dict]
    """
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))


def json_annot_loader(path_to_gt_annots, low_frame_rate_modulo, return_labels=False, old_format=False, reid=False):
    """ Converts the json files to a dictionary containing the bounding boxes
        and labels for each` frame

        Inputs:
            path_to_gt_annots   [String]
                                Specifies the path to annotation folder containing .json files
            return_labels       [boolean]
                                Sets whether the keys of the 2nd/nested dict
                                should be the object labels, instead of the tracks
            old_format          [boolean]
                                Temporary variable, sets the annotation format to the older version
            reid                [boolean]
                                Associated with the old annotation format, which evaluates for reID cases
            low_frame_rate_modulo   [int]
                                    Specifically used for simulating frame rate.
        
        Returns
            gt                  [Dict(Dict(List))]
                                - The keys of the first dict denote the frame number
                                - The keys of second dict denote the track id
                                - Finally the list contains the bounding box coordinates
    """
    # These declarations have to be cleaned up
    # labels =('s40_40_G', 's40_40_B', 'r20','motor', 'm30', 'm20_100', 'm20', 'f20_20_G',
    #             'f20_20_B', 'em_02', 'em_01', 'distance_tube', 'container_box_red',
    #             'container_box_blue', 'bearing_box_ax16', 'bearing_box_ax01', 'bearing', 'axis')
    labels = ["miscellaneous", "axis", "bearing", "bearing_box_ax01", "bearing_box_ax16",
              "container_box_blue", "container_box_red", "distance_tube", "em_01",
              "em_02", "f20_20_b", "f20_20_g", "m20", "m20_100", "m30", "motor",
              "r20", "s40_40_b", "s40_40_g"]
    label_lookup = {labels[i]: i for i in range(len(labels))}

    os.chdir(path_to_gt_annots)
    gt_files = sorted([file for file in glob.glob("frame*.json")])
    gt = {}

    if old_format:
        key = 0  # Only used with old annot format
    obj_dict = {}
    for frame in range(len(gt_files)):
        if frame % low_frame_rate_modulo != 0:
            continue
        # Load the json file for each frame
        with open(gt_files[frame]) as json_file:
            gt_frame = json.load(json_file)
        # Frame ID is the integer number of the frame
        idx = int(gt_files[frame].lstrip("frame").rstrip(".json"))
        gt[idx] = {}
        for dict_item in gt_frame['shapes']:
            if not old_format:
                if not return_labels:
                    gt[idx][int(dict_item['label'].split(" ")[-1])] = [point for sublist in dict_item['points'] for point in sublist]
                else:
                    gt[idx][int(dict_item['label'].split(" ")[0])] = [point for sublist in dict_item['points'] for point in sublist]
            else:
                if reid:
                    gt[idx][label_lookup[dict_item['label'].lower()]] = [point for sublist in dict_item['points'] for point in sublist]
                # Store each element in the list
                else:
                    if not frame:  # i.e. the first frame
                        obj_dict[key] = dict_item['label'].lower()
                        gt[idx][key] = [point for sublist in dict_item['points'] for point in sublist]
                        key += 1
                    else:
                        # Check if the same object was present in the previous frame
                        label = dict_item['label'].lower()
                        prev_obj_ids = list(gt[idx - 1].keys())
                        prev_labels = [obj_dict[obj_id] for obj_id in prev_obj_ids]
                        if label in prev_labels:
                            gt[idx][prev_obj_ids[prev_labels.index(label)]] = [point for sublist in dict_item['points']
                                                                               for point in sublist]
                        else:  # Assign it as a new object
                            obj_dict[key] = dict_item['label'].lower()
                            gt[idx][key] = [point for sublist in dict_item['points'] for point in sublist]
                            key += 1
    return gt


def get_json_results(path, filename, keys_to_int=False):
    """ Fetches the stored results from .json format and converts it to a Dictionary

        Inputs:
        path            [String]
                        Path to the results file
        filename        [String]
                        Name of the file

        Returns:
        tr_results      [Dict(Dict(List))]      MOT Format
    """
    with open('/'.join((path, filename))) as json_file:
        tr_results = json.load(json_file)
    
    if keys_to_int == True:
        tr_results = {int(frame):{track_id: bbox for track_id, bbox in tracks.items()} for frame, tracks in tr_results.items()}
    return tr_results


def convert_to_eval_format(tr_results, unique_obj_ids=True):
    """ Converts the tracking results to the same format as the ground truth data

        Inputs:
        tr_results          [Dict(Dict(List))]
                            Tracking results in MOT format
        unique_obj_ids      [boolean]
                            Set to false if you want all object IDs of a single object (label)
                            to be the same. This prevents reID evaluation and identification
                            as new object post occlusion

        Returns:
        tracks              [Dict(Dict(List))]      Standard Format
                            - The keys of the first dict denote the frame number
                            - The keys of second dict denote the object label (track id)
                            - Finally the list contains the bounding box coordinates
    """
    tracks = {}
    for obj in tr_results:  # Each track represents an object is a dict of all the frames
        for frame in tr_results[obj]:
            if int(frame) < 300:  # Temporary hack, remove this
                object_ = tr_results[obj][frame]
                # bbox = object_[:4]
                # object_id = object_[-1]
                if int(frame) not in tracks:
                    tracks[int(frame)] = {}
                if unique_obj_ids:
                    tracks[int(frame)][int(obj)] = object_[:4]
                else:
                    tracks[int(frame)][int(object_[-1])] = object_[:4]

    return tracks


def get_text_annots(file_name, frame_limit=300):
    """ Converts the output of YOLO + DeepSORT tracker to a standard format

        Inputs:
        filename        [String]
                        Name of the file
        frame_limit     [Int]
                        Number of frames to be analysed

        Returns:
        tr_results          [Dict(Dict(List))]      Standard Format
                            - The keys of the first dict denote the frame number
                            - The keys of second dict denote the object label
                            - Finally the list contains the bounding box coordinates
    """
    tr_results = {}
    with open('/'.join((os.getcwd(), file_name)), 'r') as file:
        for line in file.readlines():
            line = list(map(int, line.split()))
            if line[0] >= frame_limit:
                break
            if line[0] not in tr_results:  # Check if the frame exists in the dict
                tr_results[line[0]] = {}
            tr_results[line[0]][line[1]] = line[2:6]
    return tr_results


def check_incorrect_format(dataset):
    """
    Function to check the format of the data
    Expects the bboxes to be in the format [x1, y1, x2, y2]
    Inputs:
        dataset     [Dict(Dict(List))]      Standard Format
                    - The keys of the first dict denote the frame number
                    - The keys of second dict denote the object label
                    - Finally the list contains the bounding box coordinates
    """
    count, tlbr, brtl, bltr, trbl = 0, 0, 0, 0, 0
    for frame in dataset.keys():
        for track, coords in dataset[frame].items():
            # w, h = coords[2]-coords[0], coords[3] - coords[1]
            # if w < 0 or h < 0:
            #     if show_output:
            #         print(f"Error at frame {frame}: Track {track} width is {w} and height is {h}")
            #     else:
            #         print("Negative values detected")
            count += 1
            if coords[0] < coords[2] and coords[1] < coords[3]:
                tlbr += 1
            if coords[0] > coords[2] and coords[1] > coords[3]:
                brtl += 1
            if coords[0] < coords[2] and coords[1] > coords[3]:
                bltr += 1
            if coords[0] > coords[2] and coords[1] < coords[3]:
                trbl += 1
    print(f"Instances of format found\nTLBR: {tlbr}\nBRTL: {brtl}\nBLTR: {bltr}\nTRBL: {trbl}\nTOTAL: {count} = {tlbr+brtl+bltr+trbl}")
            

def correct_input_format(dataset):
    """
    Function to convert all bboxes in dataset to 'tlbr' format
    Expects the bboxes to be in the format [x1, y1, x2, y2]
    Inputs:
        dataset     [Dict(Dict(List))]      Standard Format
                    - The keys of the first dict denote the frame number
                    - The keys of second dict denote the object label
                    - Finally the list contains the bounding box coordinates
    """
    for frame in dataset.keys():
        for track, coords in dataset[frame].items():
            if coords[0] > coords[2] and coords[1] > coords[3]: # BRTL format
                dataset[frame][track] = [coords[2], coords[3], coords[0], coords[1]]
            elif coords[0] < coords[2] and coords[1] > coords[3]: # BLTR format
                dataset[frame][track] = [coords[0], coords[3], coords[2], coords[1]]
            elif coords[0] > coords[2] and coords[1] < coords[3]: # TRBL format
                dataset[frame][track] = [coords[2], coords[1], coords[0], coords[3]]
            else: #TLBR format
                pass


def get_mot_accum(results, gt, tlbr_to_tlwh=True):
    """ The function is called after the entire tracking process is done

        Inputs:
        results             [Dict(Dict(List))]      Standard Format
                            Results of the tracker
        gt                  [Dict(Dict(List))]      Standard Format
                            Ground Truth annotations
        tlbr_to_tlwh        [boolean]
                            If set to True, it converts the tracking bounding boxes
                            format from tlbr to tlwh i.e. 
                            [x1, y1, x2, y2] --> [x1, y1, width, height]

        Returns:
        mot_accum           [motmetrics.MOTAccumulator]
    """
    # We initialize as accumulator which stores the metrics for each frame
    mot_accum = mm.MOTAccumulator(auto_id=True)

    for frame in gt.keys():  # Sequentially cycling through the frames
        gt_ids = []
        if gt[frame]:
            gt_boxes = []
            for gt_id, box in gt[frame].items():  # the dict contains IDs and corresponding bboxes
                gt_ids.append(gt_id)
                gt_boxes.append(box)  # [x1, y1, x2, y2]

            gt_boxes = np.stack(gt_boxes, axis=0)  # Each row now represents a single bounding box
            # I guess this is done because the motmetrics library requires it in this format.
            # x1, y1, x2, y2 --> x1, y1, width, height
            gt_boxes = np.stack((gt_boxes[:, 0],
                                 gt_boxes[:, 1],
                                 gt_boxes[:, 2] - gt_boxes[:, 0],
                                 gt_boxes[:, 3] - gt_boxes[:, 1]),
                                axis=1)
        else:
            gt_boxes = np.array([])  # In case of no detections

        track_ids = []
        if results[frame]:
            track_boxes = []
            for track_id, box in results[frame].items():  # the dict contains IDs and corresponding bboxes
                track_ids.append(track_id)
                track_boxes.append(box)
            track_boxes = np.stack(track_boxes, axis=0)
            # x1, y1, x2, y2 --> x1, y1, width, height
            if tlbr_to_tlwh:
                track_boxes = np.stack((track_boxes[:, 0],
                                        track_boxes[:, 1],
                                        track_boxes[:, 2] - track_boxes[:, 0],
                                        track_boxes[:, 3] - track_boxes[:, 1]),
                                       axis=1)
        else:
            track_boxes = np.array([])

        # The max_iou represents the max_iou distance and not the overlap
        # It can be thought of as 1 - iou_overlap
        distance = mm.distances.iou_matrix(gt_boxes, track_boxes, max_iou=0.7)

        mot_accum.update(
            gt_ids,  # n
            track_ids,  # m
            distance)  # n x m

    return mot_accum


def evaluate_mot_accums(accums, names):
    """ Function to create and print the summary of the metrics
        Inputs: accums      [MOTAccumulator object]
                names       [list of strings]                
    """
    mh = mm.metrics.create()
    # summary = mh.compute_many(
    #     accums,
    #     metrics=mm.metrics.motchallenge_metrics,
    #     names=names,
    #     generate_overall=generate_overall,)
    summary = mh.compute(accums, name=names)
    # metrics=['num_frames', 'mota', 'motp'],

    str_summary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names, )
    print(str_summary)


def print_metrics(accum, name):
    """ Prints the computed metrics
        Inputs:
        accum       [motmetris.MOTAccumulator]
        name        [String]
                    Name of the tracker
    """
    mh = mm.metrics.create()
    summary = mh.compute(accum,
                         metrics=['num_frames',
                                  'mota',
                                  'motp',
                                  'idf1',
                                  'num_matches',
                                  'num_false_positives',
                                  'num_misses',
                                  'num_switches',
                                  'mostly_tracked',
                                  'partially_tracked',
                                  'mostly_lost',
                                  'num_unique_objects'],
                         name=name)
    strsummary = mm.io.render_summary(
        summary,
        formatters={'mota': '{:.2%}'.format},
        namemap={'num_frames': 'Frames',
                 'mota': 'MOTA',
                 'motp': 'MOTP',
                 'idf1': 'IDF1',
                 'num_matches': 'TP',
                 'num_false_positives': 'FP',
                 'num_misses': 'FN',
                 'num_switches': 'IDSW',
                 'mostly_tracked': 'Mostly tracked',
                 'partially_tracked': 'Partially tracked',
                 'mostly_lost': 'Mostly lost',
                 'num_unique_objects': 'Unique Objects'}
    )
    print(strsummary)


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Eval Code")
    parser.add_argument(
        "--low_frame_rate_modulo", help="Sets the number of frames to be skipped. "
        "This is used to simulate low frame rates, by skipping n frames",
        default=1, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # Load the output file    
    path_to_outputs = "/".join([os.getcwd(), 'nanonets_object_tracking/outputs'])
    # Load the tracker results
    sort_results = get_json_results(path_to_outputs, 'sort_outputs.json', keys_to_int=True)
    # pretty(sort_results)
    
    # Load the annotations
    # gt = json_annot_loader("data/annotated", old_format=True)
    gt = json_annot_loader("data/labels", args.low_frame_rate_modulo, old_format=False)
    # pretty(gt)

    # check_incorrect_format(gt)
    correct_input_format(gt)
    # check_incorrect_format(gt)

    # Accumulate the tracking results
    mot_accum = get_mot_accum(sort_results, gt)

    # Display the results
    print_metrics(mot_accum, 'SqueezeDet + SORT')
    # print(mh.list_metrics_markdown())