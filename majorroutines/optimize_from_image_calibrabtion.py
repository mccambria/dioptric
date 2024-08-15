# -*- coding: utf-8 -*-
"""
Optimize on an NV by taking an image and comparing it to a reference image

Created on April 30th, 2022

@author: mccambria
"""


# %% Imports

import pickle

import cv2
import labrad

# %% Functions
import numpy as np
from skimage.feature import match_template
from skimage.measure import ransac
from skimage.transform import AffineTransform

import majorroutines.image_sample as image_sample
import utils.image_processing as image_processing
import utils.tool_belt as tool_belt
from utils import positioning as pos


def capture_image():
    scan_range = 7
    num_steps = 10
    nv_sig = [110.0, 110.0]
    image_sample.scanning(nv_sig, scan_range, scan_range, num_steps)


def set_piezo_voltage(voltage):
    pos.set_xyz(voltage)
    return


def detect_keypoints(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors


def match_keypoints(desc1, desc2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def extract_matched_points(kp1, kp2, matches):
    points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    points2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return points1, points2


def affine_calibrate_piezo(piezo, axis, voltages, capture_func=capture_image):
    points1_list = []
    points2_list = []
    reference_image = capture_func()
    kp1, desc1 = detect_keypoints(reference_image)

    for voltage in voltages:
        set_piezo_voltage(piezo, axis, voltage)
        new_image = capture_func()
        kp2, desc2 = detect_keypoints(new_image)
        matches = match_keypoints(desc1, desc2)
        points1, points2 = extract_matched_points(kp1, kp2, matches)
        points1_list.append(points1)
        points2_list.append(points2)

    return points1_list, points2_list


def calculate_affine_transform(points1_list, points2_list):
    points1 = np.concatenate(points1_list)
    points2 = np.concatenate(points2_list)
    model, inliers = ransac(
        (points1, points2),
        AffineTransform,
        min_samples=3,
        residual_threshold=2,
        max_trials=100,
    )
    return model


def save_affine_model(model, filename):
    with open(filename, "wb") as file:
        pickle.dump(model, file)


def load_affine_model(filename):
    with open(filename, "rb") as file:
        model = pickle.load(file)
    return model


capture_image()

# if __mian__=main:

# voltage = np.linspace(-1.0, 1.0, 3)  # Example voltages from 0 to 10 V
# piezo = None  # Replace with actual piezo controller object
# axis = "x"  # Replace with the actual axis name

# # Calibration
# points1_list, points2_list = affine_calibrate_piezo(piezo, axis, voltages)
# affine_model = calculate_affine_transform(points1_list, points2_list)

# # Save the affine model
# save_affine_model(affine_model, "affine_model.pkl")

# # Load the affine model later
# loaded_affine_model = load_affine_model("affine_model.pkl")

# print(f"Affine Transformation Matrix:\n{loaded_affine_model.params}")
