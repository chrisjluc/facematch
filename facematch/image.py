import consts
import face_processing as fp

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.pyplot import imshow
from scipy.misc import imrotate
from skimage import io
from skimage.color import rgb2grey
from skimage.transform import resize


class Image(object):

    def __init__(self, f=None):
        # Could be None if we're loading an image
        # from disk, and we'll set the important
        # properties later.
        if not f:
            return

        if isinstance(f, basestring):
            # File name so read in the image
            img = io.imread(f)
        else:
            img = f

        # Normalize face angle
        face = fp.get_most_centre_face(img)
        if not face:
            raise NoFaceDetectedException()
        landmark_points = fp.get_facial_landmark_points(img, face)
        angle = fp.calculate_rotation(landmark_points)
        img = imrotate(img, -angle)

        # Normalize and crop image by centering around the facial features
        face = fp.get_most_centre_face(img)
        if not face:
            raise NoFaceDetectedException()
        landmark_points = fp.get_facial_landmark_points(img, face)
        feature_points = fp.get_facial_feature_points(img, face)

        # Estimate cropping box
        face_height = face.bottom() - face.top()
        face_width = face.right() - face.left()
        scale_height = .5
        scale_width = .5

        left = max(0, face.left() - int(face_width * scale_width))
        top = max(0, face.top() - int(face_height * scale_height))
        right = min(img.shape[1], face.right() + int(face_width * scale_width))
        bottom = min(img.shape[0], face.bottom() + int(face_height * scale_height))

        # Perform the cropping and adjust landmark / feature points
        img = img[top:bottom, left:right]
        landmark_points[:,0] -= left
        landmark_points[:,1] -= top
        feature_points[:,0] -= left
        feature_points[:,1] -= top

        # Resizing image and scaling facial landmark and feature points
        img = rgb2grey(img)
        img_shape = img.shape
        self.image = resize(img, consts.norm_shape)

        # Since we're resizing the image, we need to scale the points accordingly
        scale = np.array(consts.norm_shape).astype(float) / np.array(img_shape)
        landmark_points[:,0] = landmark_points[:,0] * scale[1]
        landmark_points[:,1] = landmark_points[:,1] * scale[0]
        feature_points[:,0] = feature_points[:,0] * scale[1]
        feature_points[:,1] = feature_points[:,1] * scale[0]

        self.landmark_points = landmark_points
        self.feature_points = feature_points

    def show(self):
        """
        For debug purposes, if this method is called within
        an ipython environment it will display the image.
        """
        self.assert_valid_state()
        imshow(self.image, cmap='Greys_r')
        coord_transpose = np.transpose(self.landmark_points)
        plt.scatter(coord_transpose[0], coord_transpose[1])
        plt.show()

    def assert_valid_state(self):
        assert(self.image is not None)
        assert(self.landmark_points is not None)
        assert(self.feature_points is not None)

class NoFaceDetectedException(Exception):
    pass
