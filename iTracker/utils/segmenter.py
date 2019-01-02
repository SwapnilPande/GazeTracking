import numpy as np
import cv2

from utils import utils
from scipy.spatial import distance as dist
from numpy import linalg as LA

# starts in left (right on person) corner and goes clockwise
reIndices = [37,38,39,40,41,42]
leIndices = [43,44,45,46,47,48]

def makeEyeMarkers(left, right, factor=1.0):
    d = np.array(dist.euclidean(left, right)) * factor * 0.2
    l = np.array(left) * factor
    r = np.array(right)* factor
    v = r - l
    norm = LA.norm(v)
    leftEyeMarks = []
    rightEyeMarks = []
    if norm > 0:
        v = v / norm
    leftEyeMarks = [l - (v*d), l + (v*d)]
    rightEyeMarks = [r - (v*d), r + (v*d)]
    return leftEyeMarks, rightEyeMarks

class Segmenter:
    def __init__(self, faceBox, leftEyeMarks, rightEyeMarks, width, height):
        self.width = width
        self.height = height
        self.faceBB = None
        self.leBB = None
        self.reBB = None
        if faceBox is not None and len(faceBox) > 0:
            self.faceBB = utils.get_square_box([int(x) for x in faceBox], [height, width])
        if leftEyeMarks is not None and len(leftEyeMarks) > 0:
            self.leBB = self.getEyeBB(leftEyeMarks)
        if rightEyeMarks is not None and len(rightEyeMarks) > 0:
            self.reBB = self.getEyeBB(rightEyeMarks)

    def makeBB(self, kp, px=0, py=0):
        if len(kp) == 1:
            s = int((self.faceBB[2] - self.faceBB[0])*0.08)
            x = [kp[0][0] - s, kp[0][0] + s]
            y = [kp[0][1] - s, kp[0][1] + s]
        else:
            x = [x[0] for x in kp]
            y = [x[1] for x in kp]
        bbox = [
            max(np.min(x) - px, 0),
            max(np.min(y) - py, 0),
            min(np.max(x) + px, self.width),
            min(np.max(y) + py, self.height)
        ]
        return utils.get_square_box([int(x) for x in bbox], [self.height, self.width])

    def getEyeBB(self, marks):
        return self.makeBB(marks, 10, 0)

    @staticmethod
    def isValidBB(bb):
        w = bb[2] - bb[0]
        h = bb[3] - bb[1]
        return bb[0] >= 0 and \
            bb[1] >= 0 and \
            w > 0 and \
            h > 0

    def isValid(self):
        return self.isValidBB(self.leBB) and \
            self.isValidBB(self.reBB) and \
            self.isValidBB(self.faceBB)

    def getSegmentBBs(self):
        return [
            self.leBB,
            self.reBB,
            self.faceBB
        ]
