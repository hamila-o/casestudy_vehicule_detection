# -*- coding: utf-8 -*-
"""
Created on Tue May 23 21:53:22 2023

@author: Oumaima
"""

import sys
import os
import scipy
import numpy as np
import argparse
import PIL
import pathlib

def IOU(bounding_box_1, bounding_box_2):
    
    x_inter_0 = max(bounding_box_1[0], bounding_box_2[0])
    y_inter_0 = max(bounding_box_1[1], bounding_box_2[1])
    x_inter_1 = min(bounding_box_1[2], bounding_box_2[2])
    y_inter_1 = min(bounding_box_1[3], bounding_box_2[3])
    intersection = 0
    if x_inter_0 >= x_inter_1 or y_inter_0 >= y_inter_1:
        intersection = 0
    else:
        intersection = (x_inter_1 - x_inter_0) * (y_inter_1 - y_inter_0)
    union = (bounding_box_1[2] - bounding_box_1[0]) * (bounding_box_1[3] - bounding_box_1[1]) + (bounding_box_2[2] - bounding_box_2[0]) * (bounding_box_2[3] - bounding_box_2[1])
    union -= intersection
    return intersection / union

def match_bboxes(bbox_gt, bbox_pred, IOU_THRESH=0.5):
    '''
    Given sets of true and predicted bounding-boxes,
    determine the best possible match.
    Parameters
    ----------
    bbox_gt, bbox_pred : N1x4 and N2x4 np array of bboxes [x1,y1,x2,y2]. 
      The number of bboxes, N1 and N2, need not be the same.
    
    Returns
    -------
    (idxs_true, idxs_pred, ious, labels)
        idxs_true, idxs_pred : indices into gt and pred for matches
        ious : corresponding IOU value of each match
        labels: vector of 0/1 values for the list of detections
    '''
    n_true = bbox_gt.shape[0]
    n_pred = bbox_pred.shape[0]
    MAX_DIST = 1.0
    MIN_IOU = 0.0

    # NUM_GT x NUM_PRED
    iou_matrix = np.zeros((n_true, n_pred))
    for i in range(n_true):
        for j in range(n_pred):
            iou_matrix[i, j] = IOU(bbox_gt[i,:], bbox_pred[j,:])

    if n_pred > n_true:
      # there are more predictions than ground-truth - add dummy rows
        diff = n_pred - n_true
        iou_matrix = np.concatenate( (iou_matrix, 
                                    np.full((diff, n_pred), MIN_IOU)), 
                                  axis=0)

    if n_true > n_pred:
      # more ground-truth than predictions - add dummy columns
          diff = n_true - n_pred
          iou_matrix = np.concatenate( (iou_matrix, 
                                    np.full((n_true, diff), MIN_IOU)), 
                                  axis=1)

    # call the Hungarian matching
    idxs_true, idxs_pred = scipy.optimize.linear_sum_assignment(1 - iou_matrix)

    if (not idxs_true.size) or (not idxs_pred.size):
        ious = np.array([])
    else:
        ious = iou_matrix[idxs_true, idxs_pred]

    # remove dummy assignments
    sel_pred = idxs_pred<n_pred
    idx_pred_actual = idxs_pred[sel_pred] 
    idx_gt_actual = idxs_true[sel_pred]
    ious_actual = iou_matrix[idx_gt_actual, idx_pred_actual]
    sel_valid = (ious_actual > IOU_THRESH)
    label = sel_valid.astype(int)

    return idx_gt_actual[sel_valid], idx_pred_actual[sel_valid], ious_actual[sel_valid], label 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default= "./data/images/", help="Input directory of the images with /")
    parser.add_argument("-o", "--output", type=str, default= "./output/", help="Output directory for the predictions with /")
    parser.add_argument("-e", "--labels", type=str, default= "./data/labels/", help="Labels directory for the evaluation. Evaluations are stored in the output directory.")
    
    args = parser.parse_args()
   
    evaluate = True
    if not os.path.exists(args.input):
        print("{} does not exist!".format(args.input))
        sys.exit(0)
    if not os.path.exists(args.labels):
        evaluate = False
        print("{} does not exist! The evaluation will not occur.".format(args.input))
    
    input_dir = args.input if args.input[-1] == '/' else (args.input + '/')
    output_dir = args.output if args.output[-1] == '/' else (args.output + '/')
    labels_dir = args.labels if args.labels[-1] == '/' else (args.labels + '/')
    
    import torch
    print('Loading pretrained YOLOv5s model...')
    model = torch.hub.load('./yolov5/', 'custom', path='yolov5s.pt', source='local')
    print('Model loaded successfully!')
    image_files = [input_dir + file_name for file_name in os.listdir(input_dir)]
    
    model.classes = [2,3,7] #Car, Motorcycle, Truck
    print('Processing images from {}...'.format(input_dir))
    results = model(image_files)
    print('All images were successfully processed!')
    if not os.path.exists(output_dir):
        print('{} does not exist. Creating the directory...'.format(output_dir))
        os.mkdir(output_dir)
    print('Saving processed images in {}...'.format(output_dir))
    results.save(save_dir=pathlib.PosixPath(output_dir))
    print('All images were successfully saved!')
    
    if(not evaluate):
        sys.exit(0)
    
    print('Evaluating the results from the labels in {}...'.format(labels_dir))
    labels_correspondances = {2:2, 3:7, 4:3} #car, truck, motorcycle

    average_iou = 0
    number_bounding_boxes = 0
    correct_predictions = 0
    number_predictions = 0
    n_files = len(image_files)
    for image_index, image_file_name in enumerate(image_files):
        
        image = np.array(PIL.Image.open(image_file_name))/255.0
        image_height, image_width, _ = image.shape
        
        label_file_name = labels_dir + image_file_name.split('/')[-1][:-4] + '.txt'
        
        true_labels = []
        with open(label_file_name, 'r') as file:
            true_labels = file.readlines()
        image_bounding_boxes = []
        for line in true_labels:
            result = line.split(' ')
            x_center, y_center, w, h = tuple(np.float32(result[1:]))
            x_0 = int(round((x_center - w / 2) * image_width))
            x_1 = int(round((x_center + w / 2) * image_width))
            y_0 = int(round((y_center - h / 2) * image_height))
            y_1 = int(round((y_center + h / 2) * image_height))
            image_bounding_boxes.append((x_0, y_0, x_1, y_1))
            
        predicted_bounding_boxes = []
        predictions = results.xywhn[image_index]
        for prediction in predictions:
            x_center = float(prediction[0])
            y_center = float(prediction[1])
            w = float(prediction[2])
            h = float(prediction[3])
            x_0 = int(round((x_center - w / 2) * image_width))
            x_1 = int(round((x_center + w / 2) * image_width))
            y_0 = int(round((y_center - h / 2) * image_height))
            y_1 = int(round((y_center + h / 2) * image_height))
            predicted_bounding_boxes.append((x_0, y_0, x_1, y_1))
        
        labels_indexes, predicted_indexes, iou, _ = match_bboxes(np.array(image_bounding_boxes), np.array(predicted_bounding_boxes), IOU_THRESH=-1)
        
        average_iou += sum(iou)
        number_bounding_boxes += len(image_bounding_boxes)
        
        for i in range(len(labels_indexes)):
            if(i < len(true_labels)):
                number_predictions += 1
                true_class = labels_correspondances[int(true_labels[labels_indexes[i]][0])]
                predicted_class = int(predictions[predicted_indexes[i]][-1])
                correct_predictions += true_class == predicted_class
        
        print('[{} / {}] images evaluated.'.format(image_index+1, n_files), end='\r')
    
    average_iou /= number_bounding_boxes
    classification_accuracy = correct_predictions / number_predictions
    saved_metric = ['Number of images: {}\n'.format(len(image_files)), 
                    'Number of bounding boxes: {}\n'.format(number_bounding_boxes), 
                    'Average Intersection Over Union (IOU): {}\n'.format(average_iou),
                    'Classification accuracy: {}\n'.format(classification_accuracy)]
    print()
    print(''.join(saved_metric))
    print('Saving metrics in {}...'.format(output_dir + 'metrics.txt'))
    with open(output_dir + 'metrics.txt', 'w') as metrics_file:
        metrics_file.writelines(saved_metric)
    print('Metrics saved successfully!')
