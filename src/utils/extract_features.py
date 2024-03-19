# pylint: disable=E
""" SOME DOCUMENTATION """

from typing import Optional

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils import build_features as bf


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocess the input image"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.filter2D(image, -1, np.ones((3, 1), np.float32) / 3)
    image = bf.stretch_image(image, k=1.5)
    image = bf.normalize_image(image, mean=0.5)

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image
