"""
author: Justin Fletcher
date: 7 Oct 2018



"""

from __future__ import absolute_import, division, print_function

import pickle
import argparse
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ObjectDetectionAnalysis(object):

    def __init__(self, truth_boxes, inferred_boxes,
                 iou_thresholds=None,
                 confidence_thresholds=None):

        # If no confidence thresholds are provided, sample some linearly.
        if iou_thresholds is None:

            iou_thresholds = np.linspace(0.01, 0.99, 5)

        # If no confidence thresholds are provided, sample some logistically.
        if confidence_thresholds is None:

            def sigmoid(x):
                return 1 / (1 + np.exp(-x))

            confidence_thresholds = sigmoid(np.linspace(-100, 100, 200))

        image_count = len(truth_boxes.keys())

        print("Image count: %d" % image_count)

        # Create a dict to map the analyses to images.
        iou_confidence_analysis = list()

        # Iterate over each image in the dataset, and evaluate performance.
        for image_number, image_name in enumerate(truth_boxes.keys()):

            if image_number % 100 == 0:

                print("Processing: %s (%d/%d)" % (image_name,
                                                  image_number,
                                                  image_count))

            # Run the analysis on this image.
            analyses = self._analyse_detections(
                truth_boxes[image_name],
                inferred_boxes[image_name],
                iou_thresholds=iou_thresholds,
                confidence_thresholds=confidence_thresholds)

            # For each IoU and confidence point, extract the counts and stats.
            for analysis in analyses:

                # Make a list image name and five always-present data values.
                data_line = [image_name,
                             analysis["iou_threshold"],
                             analysis["confidence_threshold"],
                             analysis["true_positives"],
                             analysis["false_positives"],
                             analysis["false_negatives"]]

                # # Finally, add the data line to the full matrix.
                iou_confidence_analysis.append(data_line)

        # Make the anayses a publicly-accessable attribute.
        self.iou_confidence_analysis = iou_confidence_analysis

        # Next, we build the headers.
        headers = ["image_name",
                   "iou_threshold",
                   "confidence_threshold",
                   "true_positives",
                   "false_positives",
                   "false_negatives"]

        # Build the confidence analysis into a dataframe.
        self.analysis_df = pd.DataFrame(iou_confidence_analysis,
                                        columns=headers)

    def _analyse_detections(self, truth_boxes, inferred_boxes,
                            iou_thresholds=None,
                            confidence_thresholds=None):

        # Instantiate a list to hold design-performance points.
        design_points = list()

        # Iterate over each combination of confidence and IoU threshold.
        for (iou_threshold,
             confidence_threshold) in itertools.product(iou_thresholds,
                                                        confidence_thresholds):

            # Compute the foundational detection counts at this design point.
            counts_dict = self._compute_detection_counts(truth_boxes,
                                                         inferred_boxes,
                                                         iou_threshold,
                                                         confidence_threshold)

            # Add this IoU threshold and confidence threshold to counts_dict.
            counts_dict["iou_threshold"] = iou_threshold
            counts_dict["confidence_threshold"] = confidence_threshold

            # Add this design point to the list.
            design_points.append(counts_dict)

        return(design_points)

    def _compute_detection_counts(self,
                                  truth_boxes,
                                  inferred_boxes,
                                  iou_threshold,
                                  confidence_threshold):

        # First, remove from the inferred boxes all boxes below conf threshold.
        inferred_boxes = self._filter_boxes_by_confidence(inferred_boxes,
                                                          confidence_threshold)

        # If there are no inferred boxes, all truth boxes are false negatives.
        if not inferred_boxes:

            true_postives = 0
            false_positives = 0
            false_negatives = len(truth_boxes)
            return {'true_positives': true_postives,
                    'false_positives': false_positives,
                    'false_negatives': false_negatives}

        # If there are no truth boxes, all inferred boxes are false positives.
        if not truth_boxes:
            true_postives = 0
            false_positives = len(inferred_boxes)
            false_negatives = 0
            return {'true_positives': true_postives,
                    'false_positives': false_positives,
                    'false_negatives': false_negatives}

        # Declare a list to hold inferred-truth-IoU triples.
        overlaps = list()

        # For each combination of inferred and truth boxes...
        for inferred_box, truth_box in itertools.product(inferred_boxes,
                                                         truth_boxes):

                # Compute the IoU.
                iou = self._iou(inferred_box, truth_box)

                # If the IoU exceeds the required threshold, append overlap.
                if iou > iou_threshold:

                    overlaps.append([inferred_box, truth_box, iou])

        # If no confidence-filtered boxes have enough IoU with the truth boxes.
        if not overlaps:

            # Serious mistake in original code! Reproduced here.
            true_postives = 0

            false_positives = 0

            false_negatives = len(truth_boxes)

            # Alternative, correct, definition:

            # true_postives = 0

            # false_positives = len(pred_boxes)

            # false_negatives = len(truth_boxes)

        # Otherwise, if at least one box meet IoU with the truth boxes.
        else:

            matched_inferred_boxes = list()
            matched_truth_boxes = list()

            overlaps = sorted(overlaps, key=lambda x: x[2], reverse=True)

            # Iterate over overlaps...
            for i, [inferred_box, truth_box, iou] in enumerate(overlaps):

                # ...and if neither box in this overlapping pair is matched...
                if inferred_box not in matched_inferred_boxes:

                    if truth_box not in matched_truth_boxes:

                        # ...match both by adding them to thier matched lists.
                        matched_inferred_boxes.append(inferred_box)

                        matched_truth_boxes.append(truth_box)

            # The number of true positives is the number of matched boxes.
            true_postives = len(matched_truth_boxes)

            # The number of false positives is the excess of inferrered boxes.
            false_positives = len(inferred_boxes) - len(matched_inferred_boxes)

            # The number of false negatives is the excess of truth boxes.
            false_negatives = len(truth_boxes) - len(matched_truth_boxes)

        return {'true_positives': true_postives,
                'false_positives': false_positives,
                'false_negatives': false_negatives}

    def compute_statistics(self, statistics_dict=None):

        # First, if no statisitc function dict is provided, use the defualt.
        if statistics_dict is None:

            statistics_dict = {"precision": self._precision,
                               "recall": self._recall,
                               "f1": self._f1}

        data = self.analysis_df[["true_positives",
                                 "false_positives",
                                 "false_negatives",
                                 "confidence_threshold",
                                 "iou_threshold"]]

        # Store the data in
        self.df = data

        # Sum the data over images by confidence and IoU.
        grouped = data.groupby(["iou_threshold", "confidence_threshold"]).sum()

        # Iterate over each statistic function.
        for statisitic_name, statistic_fn in statistics_dict.items():

            # Apply this statistic function across the dataframe.
            grouped[statisitic_name] = grouped.apply(statistic_fn, axis=1)

        return(grouped)

    def plot_pr_curve(self):
        '''
        Override me with better plotting.
        '''

        df = self.compute_statistics()

        # Extract unique IoU thresholds from the raw df.
        iou_thresholds = self.df["iou_threshold"].unique()

        # Start a new plot.
        ax = plt.gca()

        # Iterate over each unique IoU threshold.
        for iou_threshold in iou_thresholds:

            # Get only the rows where the IoU threshold is this IoU threshold.
            ax.scatter(df.xs(iou_threshold, level=0)["recall"],
                       df.xs(iou_threshold, level=0)["precision"],
                       label=str(iou_threshold))

        ax.set_xlabel('recall')
        ax.set_ylabel('precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        ax.set_xlim([0.0, 1.2])
        ax.set_ylim([0.0, 1.2])

        plt.show()

    def _precision(self, detection_counts_dict):
        '''
        Accepts a dict containing keys "true_positives" and "false_postives"
        and returns the precision value.
        '''

        tp = detection_counts_dict["true_positives"]
        fp = detection_counts_dict["false_positives"]

        try:

            precision = tp / (tp + fp)

        except ZeroDivisionError:

            precision = 0.0

        return(precision)

    def _recall(self, detection_counts_dict):
        '''
        Accepts a dict containing keys "true_positives" and "false_negatives"
        and returns the recall value.
        '''

        tp = detection_counts_dict["true_positives"]
        fn = detection_counts_dict["false_negatives"]

        try:

            recall = tp / (tp + fn)

        except ZeroDivisionError:

            recall = 0.0

        return(recall)

    def _f1(self, detection_counts_dict):
        '''
        Accepts a dict containing keys "true_positives", "false_positives", and
        "false_negatives" and returns the F1 score.
        '''

        precision = self._precision(detection_counts_dict)
        recall = self._recall(detection_counts_dict)

        try:

            f1 = 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:

            f1 = 0.0

        return(f1)

    def _iou(self, pred_box, gt_box):
        """Calculate IoU of single predicted and ground truth box

        Args:
            pred_box (list of floats): location of predicted object as
                [xmin, ymin, xmax, ymax]
            gt_box (list of floats): location of ground truth object as
                [xmin, ymin, xmax, ymax]

        Returns:
            float: value of the IoU for the two boxes.

        Raises:
            AssertionError: if the box is obviously malformed
        """
        x1_t, y1_t, x2_t, y2_t = gt_box
        x1_p, y1_p, x2_p, y2_p = pred_box

        if (x1_p > x2_p) or (y1_p > y2_p):
            raise AssertionError(
                "Prediction box is malformed? pred box: {}".format(pred_box))
        if (x1_t > x2_t) or (y1_t > y2_t):
            raise AssertionError(
                "Ground Truth box is malformed? true box: {}".format(gt_box))

        if (x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t):
            return 0.0

        far_x = np.min([x2_t, x2_p])
        near_x = np.max([x1_t, x1_p])
        far_y = np.min([y2_t, y2_p])
        near_y = np.max([y1_t, y1_p])

        inter_area = (far_x - near_x + 1) * (far_y - near_y + 1)
        true_box_area = (x2_t - x1_t + 1) * (y2_t - y1_t + 1)
        pred_box_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
        iou = inter_area / (true_box_area + pred_box_area - inter_area)
        return iou

    def _filter_boxes_by_confidence(self,
                                    inferred_boxes,
                                    confidence_threshold):

        filtered_boxes = list()

        # Iterate over each inferred box...
        for inferred_box in inferred_boxes:

            # ...check if the confidence of this box exceeds the threshold...
            if inferred_box["confidence"] > confidence_threshold:

                # If it does, add it to the filtered boxes, which are returned.
                filtered_boxes.append(inferred_box["box"])

        return(filtered_boxes)


def loadDetectionsDataFromPickle(pickle_file):

    print('Loading pickle file...')

    with open(pickle_file, 'rb') as handle:

        unserialized_data = pickle.load(handle)

        handle.close()

        detections_dict = unserialized_data

        return detections_dict


def extract_boxes_frcnn(detections_dict, score_limit=0.0):
    '''
    Extracts boxes from a given detection dict. Rewrite this for different
    detection dictionary storing architectures.
    '''

    inferred_boxes = dict()

    # Zip over the inferred dectections dict values.
    for image_name, boxes, scores in zip(detections_dict['image_name'],
                                         detections_dict['detection_boxes'],
                                         detections_dict['detection_scores']):

        scored_boxes = list()

        # Iterate over the list of inferred boxes and scores...
        for box, score in zip(boxes, scores):

            # ...and if the score exceeds the limit...
            if score >= score_limit:

                # ....create a mapping dict, and append it to the list.
                scored_box_dict = {"box": box,
                                   "confidence": score}

                scored_boxes.append(scored_box_dict)

        # Finally, map the scored boxes list to the image name.
        inferred_boxes[image_name] = scored_boxes

    truth_boxes = dict()

    # Iterate over the truth box dict values.
    for image_name, boxes in zip(detections_dict['image_name'],
                                 detections_dict['ground_truth_boxes']):

        # Map each list of boxes to the cooresponding image name.
        truth_boxes[image_name] = boxes

    return(inferred_boxes, truth_boxes)


def extract_boxes_yolo3(detections_dict, score_limit=0.0):
    '''
    Extracts boxes from a given detection dict. Rewrite this for different
    detection dictionary storing architectures.
    '''

    inferred_boxes = dict()

    # Zip over the inferred dectections dict values.
    for image_name, boxes, in zip(detections_dict['image_name'],
                                  detections_dict['detection_boxes']):

        scored_boxes = list()

        # Iterate over the list of inferred boxes and scores...
        for box in boxes:

            if box:

                score = box[4]

                # ...and if the score exceeds the limit...
                if score >= score_limit:

                    # ....create a mapping dict, and append it to the list.
                    scored_box_dict = {"box": box[0:4],
                                       "confidence": score}

                    scored_boxes.append(scored_box_dict)

        # Finally, map the scored boxes list to the image name.
        inferred_boxes[image_name] = scored_boxes

    truth_boxes = dict()

    # Iterate over the truth box dict values.
    for image_name, boxes in zip(detections_dict['image_name'],
                                 detections_dict['ground_truth_boxes']):

        classless_boxes = list()

        for box in boxes:

            if box:

                classless_boxes.append([float(b) for b in box[0:4]])

        # Map each list of boxes to the cooresponding image name.
        truth_boxes[image_name] = classless_boxes

    return(inferred_boxes, truth_boxes)


def main(unused_argv):

    detections_dict = loadDetectionsDataFromPickle(FLAGS.pickle_file)

    if FLAGS.pickle_flavor is "frcnn":

        (inferred_boxes,
         truth_boxes) = extract_boxes_frcnn(detections_dict,
                                            score_limit=FLAGS.score_limit)

    elif FLAGS.pickle_flavor is "yolo3":

        (inferred_boxes,
         truth_boxes) = extract_boxes_yolo3(detections_dict,
                                            score_limit=FLAGS.score_limit)

    else:

        print(FLAGS.pickle_flavor + " is not a recognized pickle flavor!")
        return(1)

    # for i, key in enumerate(truth_boxes.keys()):
    #     print("----------------")
    #     print(inferred_boxes[key])
    #     print(truth_boxes[key])

    #     if i is 100:

    #         explode

    # Run the analysis.
    detection_analysis = ObjectDetectionAnalysis(truth_boxes, inferred_boxes)

    # Compute the statistics.
    stat_df = detection_analysis.compute_statistics()

    if FLAGS.print_dataframe:

        # Display the dataframe.
        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None):
            print(stat_df)

    if FLAGS.plot_pr_curve:

        # Plot the PR curve.
        detection_analysis.plot_pr_curve()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # parser.add_argument("--pickle_file",
    #                     default='..\\object_detection_analysis\\object_detection_analysis\\detections_faster_rcnn_e9_templog0.pickle',
    #                     help="The pickle file to use.")

    # parser.add_argument("--pickle_flavor",
    #                     default="frcnn",
    #                     help="One of [frcnn, yolo3]. Indicates pickle format.")

    parser.add_argument("--pickle_file",
                        default= '..\\object_detection_analysis\\object_detection_analysis\\yolo3_detections.pickle',
                        help="The pickle file to use.")

    parser.add_argument("--pickle_flavor",
                        default="yolo3",
                        help="One of [frcnn, yolo3]. Indicates pickle format.")

    parser.add_argument("--score_limit",
                        default=0.01,
                        help="All inferred boxes w/ lower scores are removed.")

    parser.add_argument("--print_dataframe",
                        default=True,
                        help="If True, prints a pandas dataframe of analysis.")

    parser.add_argument("--plot_pr_curve",
                        default=True,
                        help="If True, plots the PR curve.")

    FLAGS, unparsed = parser.parse_known_args()

    main(unparsed)
