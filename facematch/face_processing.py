import consts

import dlib
import math
import numpy as np
from scipy.spatial import distance
import sys

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(consts.face_predictor_path)

def _get_centroid(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    return (sum(x) / len(points), sum(y) / len(points))

def get_most_centre_face(image):
    height = image.shape[0]
    width = image.shape[1]
    image_centre = width / 2, height / 2
    most_centre_face = None
    most_centre_dist = sys.maxint

    detected_faces = detector(image, 1)
    for d in detected_faces:
        face_centre = (d.right() + d.left()) / 2, (d.bottom() + d.top()) / 2
        centre_dist = distance.euclidean(image_centre, face_centre)
        if centre_dist < most_centre_dist:
            most_centre_dist = centre_dist
            most_centre_face = d

    return most_centre_face

def get_facial_landmark_points(image, face, w=.6, eye_scale=.05):
    points = np.array([[p.x, p.y] for p in predictor(image, face).parts()])
    landmarks = [None for _ in range(9)]

    p = _get_centroid([points[19], points[24]])
    dx = p[0] - points[51][0]
    dy = p[1] - points[51][1]

    # Left
    landmarks[0] = points[19]

    p = _get_centroid(points[40:42])
    landmarks[1] = (p[0] - dx * eye_scale, p[1] - dy * eye_scale)

    landmarks[2] = (points[2][0] * (1 - w) + w * points[31][0],
                      points[2][1] * (1 - w) + w * points[31][1])

    # Centre
    p = _get_centroid([points[19], points[24]])
    landmarks[3] = (p[0] + dx * .1, p[1] + dy * .1)
    landmarks[4] = points[29]
    landmarks[5] = points[51]

    # Right
    landmarks[6] = points[24]

    p = _get_centroid(points[46:48])
    landmarks[7] = (p[0] - dx * eye_scale, p[1] - dy * eye_scale)

    landmarks[8] = (points[14][0] * (1 - w) + w * points[35][0],
                      points[14][1] * (1 - w) + w * points[35][1])

    return np.array(landmarks)

def get_facial_feature_points(image, face):
    points = np.array([[p.x, p.y] for p in predictor(image, face).parts()])
    facial_feature_points = [None for i in range(5)]

    # Left and right eyes, nose, left and right points of the mouth
    facial_feature_points[0] = _get_centroid(np.concatenate((points[37:39], points[40:42])))
    facial_feature_points[1] = _get_centroid(np.concatenate((points[43:45], points[46:48])))
    facial_feature_points[2] = points[30]
    facial_feature_points[3] = points[48]
    facial_feature_points[4] = points[54]

    return np.array(facial_feature_points)

def calculate_rotation(landmark_points):
    """
    Return degrees of how much the face is rotated off the vertical axis
    """
    u = landmark_points[3]
    v = landmark_points[5]
    return math.degrees(np.arctan((v[0] - u[0]) / (v[1] - u[1])))
