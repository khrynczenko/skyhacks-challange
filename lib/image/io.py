import cv2
import numpy as np


def save_image(image: np.ndarray, path: str):
    if image.ndim == 3:
        if image.dtype in [np.uint8, np.uint16] and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)


def read_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    # opencv by default stores RGB value in reverse i.e BGR. Below if converts back to RGB representation.
    if image.ndim == 3:
        if image.dtype in [np.uint8, np.uint16] and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
