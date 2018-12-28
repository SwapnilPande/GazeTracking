import os
from os_detector import detect_os, isWindows
detect_os()

import json
import numpy as np
import cv2

import utils
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
        self.faceGrid = None
        self.eyeGrid = None
        if faceBox is not None and len(faceBox) > 0:
            self.faceBB = utils.get_square_box([int(x) for x in faceBox], [height, width])
            self.faceGrid = self.getFaceGrid()
        if leftEyeMarks is not None and len(leftEyeMarks) > 0:
            self.leBB = self.getEyeBB(leftEyeMarks)
        if rightEyeMarks is not None and len(rightEyeMarks) > 0:
            self.reBB = self.getEyeBB(rightEyeMarks)
        if self.leBB is not None and self.reBB is not None:
            self.eyeGrid = self.getEyeGrid()

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

    def getEyeGrid(self):
        # make sure the facegrid is square
        # (pad with zeroes on each side)
        size = max(self.height, self.width)
        #Create array of zeros
        eyeGrid = np.zeros((size, size))
        diff = self.height - self.width
        ox = 0
        oy = 0
        # compute offsets from squaring
        if diff > 0: # height > width
            ox = int(abs(diff) / 2)
        elif diff < 0: # height < width
            oy = int(abs(diff) / 2)
        # get the left eye bounding box
        bb = self.leBB
        # make sure to use any offsets from making the image square
        x = int(bb[0] + ox)
        y = int(bb[1] + oy)
        w = int(bb[2] - bb[0])
        h = int(bb[3] - bb[1])

        xBound = int(x+w)
        yBound = int(y+h)

        if(x < 0):
            x = 0
        if(y < 0):
            y = 0
        if(xBound > size):
            xBound = size
        if(yBound > size):
            yBound = size

        for i in range(x,xBound):
            for j in range(y,yBound):
                eyeGrid[j][i] = 1
        # get the right eye bounding box
        bb = self.reBB
        # make sure to use any offsets from making the image square
        x = int(bb[0] + ox)
        y = int(bb[1] + oy)
        w = int(bb[2] - bb[0])
        h = int(bb[3] - bb[1])

        xBound = int(x+w)
        yBound = int(y+h)

        if(x < 0):
            x = 0
        if(y < 0):
            y = 0
        if(xBound > size):
            xBound = size
        if(yBound > size):
            yBound = size

        for i in range(x,xBound):
            for j in range(y,yBound):
                eyeGrid[j][i] = 1

        # now return the eye grid
        return eyeGrid

    def getFaceGrid(self):
        # make sure the facegrid is square
        # (pad with zeroes on each side)
        size = max(self.height, self.width)
        #Create array of zeros
        faceGrid = np.zeros((size, size))
        diff = self.height - self.width
        ox = 0
        oy = 0
        # compute offsets from squaring
        if diff > 0: # height > width
            ox = int(abs(diff) / 2)
        elif diff < 0: # height < width
            oy = int(abs(diff) / 2)
        # get the face bounding box
        bb = self.faceBB
        # make sure to use any offsets from making the image square
        x = int(bb[0] + ox)
        y = int(bb[1] + oy)
        w = int(bb[2] - bb[0])
        h = int(bb[3] - bb[1])

        xBound = int(x+w)
        yBound = int(y+h)

        if(x < 0):
            x = 0
        if(y < 0):
            y = 0
        if(xBound > size):
            xBound = size
        if(yBound > size):
            yBound = size

        # set faceGrid bounding box (in 25x25 shape)
        factor = 25 / size
        self.faceGridBB = [
            int(x * factor), int(y * factor),
            int(xBound * factor), int(yBound * factor)
        ]

        for i in range(x,xBound):
            for j in range(y,yBound):
                faceGrid[j][i] = 1
        return faceGrid

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
            self.isValidBB(self.faceBB) and \
            self.isValidBB(self.faceGridBB)

    def getSegmentJSON(self):
        if not self.isValid():
            return None
        return {
            'leftEye': self.leBB,
            'rightEye': self.reBB,
            'face': self.faceBB,
            'faceGrid': self.faceGrid,
            'faceGridBB': self.faceGridBB
        }

    def getSegmentBBs(self):
        return [
            self.leBB,
            self.reBB,
            self.faceBB
        ]

class Subject:
    def __init__(self, path):
        self.path = path
        # output json structures
        # these will be read in
        self.framesJSON = {}
        self.dotJSON = {}
        # these will be created
        self.leftEyeJSON = {
            'X': [],
            'Y': [],
            'W': [],
            'H': [],
            'IsValid': []
        }
        self.rightEyeJSON = {
            'X': [],
            'Y': [],
            'W': [],
            'H': [],
            'IsValid': []
        }
        self.faceJSON = {
            'X': [],
            'Y': [],
            'W': [],
            'H': [],
            'IsValid': []
        }
        self.faceGridJSON = {
            'X': [],
            'Y': [],
            'W': [],
            'H': [],
            'IsValid': []
        }
        self.poseJSON = {
            'Confidence': [],
            'HeadPose': [],
            'Markers5': [],
            'Markers68': []
        }

    def addPose(self, index, confidence=0.0, pose=None, markers5=None, markers68=None):
        self.poseJSON['Confidence'].append(float(confidence))
        self.poseJSON['HeadPose'].append([])
        self.poseJSON['Markers5'].append([])
        self.poseJSON['Markers68'].append([])
        if pose is not None:
            self.poseJSON['HeadPose'][index] = pose
        if markers5 is not None:
            self.poseJSON['Markers5'][index] = markers5
        if markers68 is not None:
            self.poseJSON['Markers68'][index] = markers68

    def addSegments(self, index, segmentJSON=None):
        # Note: this function does not update dotJSON or framesJSON -
        #       since they are loaded and should be unchanged
        self.leftEyeJSON['X'].append(0)
        self.leftEyeJSON['Y'].append(0)
        self.leftEyeJSON['W'].append(0)
        self.leftEyeJSON['H'].append(0)
        self.leftEyeJSON['IsValid'].append(0)
        self.rightEyeJSON['X'].append(0)
        self.rightEyeJSON['Y'].append(0)
        self.rightEyeJSON['W'].append(0)
        self.rightEyeJSON['H'].append(0)
        self.rightEyeJSON['IsValid'].append(0)
        self.faceJSON['X'].append(0)
        self.faceJSON['Y'].append(0)
        self.faceJSON['W'].append(0)
        self.faceJSON['H'].append(0)
        self.faceJSON['IsValid'].append(0)
        self.faceGridJSON['X'].append(0)
        self.faceGridJSON['Y'].append(0)
        self.faceGridJSON['W'].append(0)
        self.faceGridJSON['H'].append(0)
        self.faceGridJSON['IsValid'].append(0)
        if segmentJSON is not None:
            # update leftEyeJSON
            le = segmentJSON["leftEye"]
            self.leftEyeJSON['X'][index] = le[0]
            self.leftEyeJSON['Y'][index] = le[1]
            self.leftEyeJSON['W'][index] = le[2] - le[0]
            self.leftEyeJSON['H'][index] = le[3] - le[1]
            self.leftEyeJSON['IsValid'][index] = 1
            # update rightEyeJSON
            re = segmentJSON["rightEye"]
            self.rightEyeJSON['X'][index] = re[0]
            self.rightEyeJSON['Y'][index] = re[1]
            self.rightEyeJSON['W'][index] = re[2] - re[0]
            self.rightEyeJSON['H'][index] = re[3] - re[1]
            self.rightEyeJSON['IsValid'][index] = 1
            # update faceJSON
            f = segmentJSON["face"]
            self.faceJSON['X'][index] = f[0]
            self.faceJSON['Y'][index] = f[1]
            self.faceJSON['W'][index] = f[2] - f[0]
            self.faceJSON['H'][index] = f[3] - f[1]
            self.faceJSON['IsValid'][index] = 1
            # update faceGridJSON
            # Note: FG is 1-indexed, so we must add one
            fg = segmentJSON["faceGridBB"]
            self.faceGridJSON['X'][index] = fg[0] + 1
            self.faceGridJSON['Y'][index] = fg[1] + 1
            self.faceGridJSON['W'][index] = fg[2] - fg[0]
            self.faceGridJSON['H'][index] = fg[3] - fg[1]
            self.faceGridJSON['IsValid'][index] = 1

    def writeSegmentFiles(self, folder):
        fullDir = self.path + '/' + folder
        # check if the folder exists
        if not os.path.isdir(fullDir):
            os.mkdir(fullDir)
        # write frames.json
        fname = 'frames.json'
        with open(fullDir + '/' + fname, 'w') as f:
            f.write(json.dumps(self.framesJSON))
        # write appleFace.json
        fname = 'appleFace.json'
        with open(fullDir + '/' + fname, 'w') as f:
            f.write(json.dumps(self.faceJSON))
        # write appleLeftEye.json
        fname = 'appleLeftEye.json'
        with open(fullDir + '/' + fname, 'w') as f:
            f.write(json.dumps(self.leftEyeJSON))
        # write appleRightEye.json
        fname = 'appleRightEye.json'
        with open(fullDir + '/' + fname, 'w') as f:
            f.write(json.dumps(self.rightEyeJSON))
        # write faceGrid.json
        fname = 'faceGrid.json'
        with open(fullDir + '/' + fname, 'w') as f:
            f.write(json.dumps(self.faceGridJSON))
        # write dotInfo.json
        fname = 'dotInfo.json'
        with open(fullDir + '/' + fname, 'w') as f:
            f.write(json.dumps(self.dotJSON))
        # write pose.json
        fname = 'pose.json'
        with open(fullDir + '/' + fname, 'w') as f:
            f.write(json.dumps(self.poseJSON))

    def getFramesJSON(self):
        with open(self.path + '/frames.json') as f:
            frames = json.load(f)
        # update frameJSON to be same as loaded file
        self.framesJSON = frames
        return frames

    def getFaceJSON(self):
        with open(self.path + '/appleFace.json') as f:
            faceMeta = json.load(f)
        return faceMeta

    def getEyesJSON(self):
        with open(self.path + '/appleLeftEye.json') as f:
            leftEyeMeta = json.load(f)
        with open(self.path + '/appleRightEye.json') as f:
            rightEyeMeta = json.load(f)
        return (leftEyeMeta, rightEyeMeta)

    def getFaceGridJSON(self):
        with open(self.path + '/faceGrid.json') as f:
            faceGridMeta = json.load(f)
        return faceGridMeta

    def getDotJSON(self):
        with open(self.path + '/dotInfo.json') as f:
            dotMeta = json.load(f)
        # update dotJSON to be the same as the loaded file
        self.dotJSON = dotMeta
        return dotMeta

    def getImage(self, imagePath):
        return cv2.imread(imagePath)

def main():
    import sys
    import imutils
    from imutils import face_utils
    from functools import reduce
    import dlib
    import time
    import argparse
    import queue as Q
    from multiprocessing import Process, Queue
    import threading
    from tqdm import tqdm

    # parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input-folder", type=str, default=".",
                    help="Folder containing unzipped test/train/validate folders (which contain subjects)")
    ap.add_argument("-o", "--output-prefix", type=str, default="custom_segmentation",
                    help="Name / Prefix for output folder")
    ap.add_argument("-d", "--debug", action="store_true", default=False,
                    help="Flag to enable debug output")
    ap.add_argument("-c", "--use-confidence", action="store_true", default=False,
                    help="Flag to enable use of confidence as 'IsValid' metric")
    ap.add_argument("-t", "--confidence-threshold", type=float, default=0.7,
                    help="Number of threads to spawn")
    ap.add_argument("-n", "--num-threads", type=int, default="10",
                    help="Number of threads to spawn")
    args = vars(ap.parse_args())

    DEBUG = args["debug"]

    # init shared variables
    output_prefix = args["output_prefix"]
    conf_threshold = args["confidence_threshold"]
    use_confidence = args["use_confidence"]
    detectorWidth = 400
    faceBoxScale = 0.15

    def process_subject(thread_index, processed_queue, done_queue, sub_queue):
        """Get subject from subject queue. This function is used for multiprocessing"""
        # init process/thread variables
        if DEBUG:
            print("[THREAD] loading facial landmark predictor...")
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('../model/shape_predictor_5_face_landmarks.dat')
        timeout = 1 # 1 second timeout waiting for more subjects
        while True:
            try:
                # get subject from queue
                subject = sub_queue.get(timeout=timeout)
            except Q.Empty as inst:
                break;
            subjectPath = subject.path
            subjectID = subjectPath.split('/')[-1]

            if DEBUG:
                print("[THREAD] Processing subject:",subjectID)

            # load MIT metadata
            frameNames = subject.getFramesJSON()
            # Collecting metadata about face, eyes, facegrid, labels
            face = subject.getFaceJSON()
            leftEye, rightEye = subject.getEyesJSON()
            faceGrid = subject.getFaceGridJSON()
            dotInfo = subject.getDotJSON()


            frameBar = tqdm(total=len(frameNames),
                            unit="Frames",
                            desc="Subject {}".format(subjectID),
                            position=thread_index,
                            leave=False
            )
            barStep = 10

            # Iterate over frames for the current subject
            for index, (frame, fv, lv, rv, fgv) in enumerate(zip(frameNames,
                                                             face['IsValid'],
                                                             leftEye['IsValid'],
                                                             rightEye['IsValid'],
                                                             faceGrid['IsValid'])):
                # we'll need to make sure all frames are processed so
                # we must call Subject::addSegments for every frame -
                # it will set IsValid to False if segmentJSON is None
                segmentJSON = None
                pose = None
                markers5 = None
                markers68 = None
                confidence = 0.0
                # Check if cur frame is valid
                if(use_confidence or fv*lv*rv*fgv == 1):
                    # Generate path for frame
                    framePath = subjectPath + "/frames/" + frame
                    # load image data
                    image = subject.getImage(framePath)
                    originalWidth = image.shape[1]
                    factor = originalWidth / detectorWidth
                    frame = imutils.resize(image, width=detectorWidth)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    dets, scores, idx = detector.run(gray, 0)

                    if dets is not None and len(dets) > 0:
                        facebox = dets[0]
                        confidence = scores[0]
                        try:
                            shape = predictor(gray, facebox)
                            shape = face_utils.shape_to_np(shape)
                            # loop over the (x, y)-coordinates for the facial landmarks
                            # and draw each of them
                            leftEyeMarks = []
                            rightEyeMarks = []
                            markers5 = []
                            for (i, (x, y)) in enumerate(shape):
                                [x,y] = [int(x*factor),int(y*factor)]
                                markers5.append([x,y])
                                if i == 0 or i ==1:
                                    leftEyeMarks.append([x,y])
                                if i == 2 or i ==3:
                                    rightEyeMarks.append([x,y])

                            # segment the image based on markers and facebox
                            facebox = utils.dlib_to_box(facebox, factor, faceBoxScale)
                            seg = Segmenter(
                                facebox,
                                leftEyeMarks,
                                rightEyeMarks,
                                image.shape[1],
                                image.shape[0]
                            )
                            segmentJSON = seg.getSegmentJSON()
                        except cv2.error as inst:
                            print("[THREAD] Error processing subject:",
                                  subjectID,'frame:', index, inst)
                # add segment data to subject
                subject.addSegments(index, segmentJSON)
                subject.addPose(index, confidence, pose, markers5, markers68)
                if index > 0 and (index % barStep) == 0:
                    frameBar.update(barStep)

            # write out the metadata file
            subject.writeSegmentFiles(output_prefix)
            frameBar.close()

            if DEBUG:
                print("[THREAD] Finished processing subject:", subjectID)
            # write how many frames we processed
            processed_queue.put(len(frameNames))

        # mark that we're done here!
        if DEBUG:
            print("[THREAD] subject processing done!")
        done_queue.put(True)

    # get directory to subjects
    import glob
    base_path = args["input_folder"]
    train_path = base_path + '/train/*'
    test_path = base_path + '/test/*'
    validate_path = base_path + '/validate/*'
    subjectDirs = [
        glob.glob(train_path),
        glob.glob(test_path),
        glob.glob(validate_path)
    ]
    # flatten the directories
    subjectDirs = [item for sublist in subjectDirs for item in sublist]
    num_subjects = len(subjectDirs)
    num_subjects_processed = 0

    # set up the progress bar
    pbar = tqdm(total=num_subjects,
                unit="Subjects",
                desc="Subjects processed",
                position=0
    )

    # TODO: find better way to control multiprocessing and memory usage
    parallelization = min(args["num_threads"], num_subjects)

    # Setup process and queues for multiprocessing.
    sub_queue = Queue()
    done_queue = Queue()
    processed_queue = Queue()
    # spawn some number of threads / processes here
    tids = []
    if isWindows():
        for tid in range(parallelization):
            thread = threading.Thread(target=process_subject,
                                      args=(tid+1, processed_queue, done_queue, sub_queue))
            thread.daemon = True
            thread.start()
            tids.append(thread)
    else:
        for tid in range(parallelization):
            box_process = Process(target=process_subject,
                                  args=(tid+1, processed_queue, done_queue, sub_queue))
            box_process.start()
            tids.append(box_process)


    # TODO: might need better control over memory management
    max_queue_size = parallelization * 2
    while True:
        if sub_queue.qsize() < max_queue_size and num_subjects_processed < num_subjects:
            subDir = subjectDirs[num_subjects_processed]
            subject = Subject(subDir)
            # feed subject into subject queue.
            sub_queue.put(subject)
            # update the number of subjects we have processed
            num_subjects_processed += 1
        else:
            # wait to not take up cpu time
            time.sleep(0.1)
        pbar.update(processed_queue.qsize() - pbar.n)
        # are the threads done?
        if done_queue.qsize() >= len(tids):
            print("[INFO] All threads done, exiting!")
            break;

    # clean up process
    pbar.close()
    if not isWindows():
        for tid in tids:
            tid.terminate()
            tid.join()


if __name__ == '__main__':
    main()
