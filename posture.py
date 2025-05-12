# Angela Zhang
# Webcam Posture Detector
# posture.py
# Created to calculate head tilt angles and determine if the user has bad posture

import numpy as np
import cv2

def calculate_head_tilt_angles(forehead, chin, image=None):
    dx = forehead[0] - chin[0] # horizontal distance between chin and forehead
    dy = chin[1] - forehead[1] # vertical distance between chin and forehead
    dz = (forehead[2] - chin[2]) * image.shape[0] # front back difference (scaled for size)

    # side tilt angle
    side_radians = np.arctan2(dx, dy)  # uses horizontal and vertical distance

    # up down tilt using depth
    up_down_radians = np.arctan2(dz, dy) # uses depth and vertical distance
    
    side_angle = np.degrees(side_radians) # convert to degrees
    up_down_angle = np.degrees(up_down_radians) # convert to degrees


    return side_angle, up_down_angle


def is_bad_posture(side_angle, up_down_angle, side_threshold=15, up_threshold=10, down_threshold=5):
    side_tilt_bad = abs(side_angle) > side_threshold # side tilt bad if angle is greater than threshold

    # use different thresholds based on whether user is looking up or down
    if up_down_angle > 0:
        vertical_tilt_bad = up_down_angle > up_threshold  # looking up
    else:
        vertical_tilt_bad = abs(up_down_angle) > down_threshold  # looking down


    return side_tilt_bad or vertical_tilt_bad

