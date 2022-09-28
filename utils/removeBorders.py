import cv2
import numpy as np
import os
from pathlib import Path

def crop(image):
    y_nonzero, x_nonzero, _ = np.nonzero(image>25) ## arbitrary number incase pixels are not pitch black (0)
    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

ROOT = Path(__file__).parent.parent
DATA = ROOT / 'data/cfp/'
for image in os.listdir(DATA):
    img = cv2.imread(str(DATA) + '/' + image)
    cropped = crop(img)
    cv2.imwrite('data/cropped/'+image, cropped)

