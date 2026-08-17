import cv2
import numpy as np


class ImagePreprocessor:

    def preprocess(self, image):

        if image is None:
            raise ValueError("Image is None")

        if not isinstance(image, np.ndarray):
            raise TypeError("Expected numpy.ndarray")

        return image