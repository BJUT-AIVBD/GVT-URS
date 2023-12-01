from ..builder import PIPELINES
import cv2
import numpy as np


@PIPELINES.register_module()
class EqHist(object):
    def __init__(self):
        self.clipLimit = 2.0
        self.tileGridSize = (8, 8)

    def __call__(self, results):
        gray = cv2.cvtColor(results['img'], cv2.COLOR_RGB2GRAY)

        clahe = cv2.createCLAHE(self.clipLimit, self.tileGridSize)
        gray = (gray*255).astype(np.uint8)
        clahe.apply(gray, gray)
        gray = (gray.astype(np.float32))/255
        results['img'] = np.dstack((results['img'], gray))
        return results
