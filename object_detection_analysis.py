"""
author: Justin Fletcher
date: 7 Oct 2018



"""

from __future__ import absolute_import, division, print_function

import itertools

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle


class ObjectDetectionAnalysis(object):

    def __init__(self, truth_boxes, inferred_boxes,
                 confidence_thresholds=None):

        print("Image count: %d" % len(truth_boxes.keys()))

        # Create a dict to map the analyses to images.
        iou_confidence_analysis = list()

        # Iterate over each image in the dataset, and evaluate performance.
        for image_name in truth_boxes.keys():

            print("Processing: %s" % image_name)

            # start = timer()

            analyses = self._analyse_detections(truth_boxes[image_name],
                                                inferred_boxes[image_name])
            # end = timer()
            # print(end - start)

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
                            confidence_thresholds=None):

        iou_thresholds = np.linspace(0.5, 0.90, 3)

        if confidence_thresholds is None:

            def sigmoid(x):
                return 1 / (1 + np.exp(-x))

            confidence_thresholds = sigmoid(np.linspace(-10, 10, 100))

            # print(confidence_thresholds)

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

        all_pred_indices = range(len(inferred_boxes))
        all_gt_indices = range(len(truth_boxes))

        if len(all_pred_indices) == 0:
            true_postives = 0
            false_positives = 0
            false_negatives = len(truth_boxes)
            return {'true_positives': true_postives,
                    'false_positives': false_positives,
                    'false_negatives': false_negatives}

        if len(all_gt_indices) == 0:
            true_postives = 0
            false_positives = len(inferred_boxes)
            false_negatives = 0
            return {'true_positives': true_postives,
                    'false_positives': false_positives,
                    'false_negatives': false_negatives}

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

        # Sum the data over images by confidence and IoU.
        grouped = data.groupby(["confidence_threshold", "iou_threshold"]).sum()

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

        # Extract IoU thresholds.
        iou_thresholds = df.iou_threshold.unique()

        ax = plt.gca()

        for iou_threshold in iou_thresholds:

            ax.scatter(df["recall"], df["precision"], label=iou_threshold)

        ax.set_xlabel('recall')
        ax.set_ylabel('precision')
        ax.set_title('Precision-Recall curve for')
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

COLORS = [
    '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
    '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
    '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
    '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']


def loadDetectionsDataFromPickle(pickle_file):

    print('Loading pickle file...')

    with open(pickle_file, 'rb') as handle:

        unserialized_data = pickle.load(handle)

        handle.close()

        detections_dict = unserialized_data

        return detections_dict


def extract_boxes(detections_dict, score_limit=0.0):
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


def plot_pr_curve(precisions,
                  recalls,
                  category='Satellite',
                  label=None,
                  color=None,
                  ax=None):
    """Simple plotting helper function"""

    if ax is None:
        plt.figure(figsize=(10, 8))
        ax = plt.gca()

    if color is None:
        color = COLORS[0]
    ax.scatter(recalls, precisions, label=label, s=20, color=color)
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.set_title('Precision-Recall curve for {}'.format(category))
    ax.set_xlim([0.0, 1.3])
    ax.set_ylim([0.0, 1.2])
    return ax


if __name__ == "__main__":

    pickle_file = 'detections.pickle'
    detections_dict = loadDetectionsDataFromPickle(pickle_file)

    # These dicts share a key structure.
    inferred_boxes, truth_boxes = extract_boxes(detections_dict,
                                                score_limit=0.01)

    # Run the analysis.
    detection_analysis = ObjectDetectionAnalysis(truth_boxes, inferred_boxes)

    # Compute the statistics.
    stat_df = detection_analysis.compute_statistics()

    # Display the dataframe.
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None):
        print(stat_df)

    # Plot the PR curve.
    detection_analysis.plot_pr_curve()
