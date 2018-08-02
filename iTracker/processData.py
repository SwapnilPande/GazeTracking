import json
import numpy as np
import cv2
import random
import os
import shutil
import math
import tarfile
from keras.utils import Sequence
from keras.utils.training_utils import multi_gpu_model
from uiUtils import yesNoPrompt, createProgressBar
import imageUtils


def initializeData(pathToData, pathTemp, trainProportion, validateProportion, args):
	#Calculating test data prooportion based on size of data
	testProportion = 1 - trainProportion - validateProportion

	#Getting all of the subject directories in the data directories
	dataDirs = os.listdir(path=pathToData)

	subjectDirs = []
	#Create list containing all of the items ending with .zip
	#This represents the zip files containing data for each subject
	print('Locating data')
	for item in dataDirs:
		if item.endswith('.tar.gz'):
			subjectDirs.append(pathToData + '/' + item)

	#Store number of subjects in variable
	numSubjects = len(subjectDirs) ##TODO Decide if needded
	print('Found ' + str(numSubjects) + " subjects (zipped)")
	print()

	#Create data to store unzipped subject data
	#Create three subdirectories to split train, validate, and test data
	print('Creating temporary directory to store unzipped data')
	tempDataDir = pathTemp + '/temp'
	trainDir = tempDataDir + '/train'
	validateDir = tempDataDir + '/validate'
	testDir = tempDataDir + '/test'

	#Number of training, validation, and test subjects
	numTrainSubjects = round(trainProportion*numSubjects)
	numValidateSubjects = round(validateProportion*numSubjects)
	numTestSubjects = numSubjects - numTrainSubjects - numValidateSubjects
	
	#Flag to store whether or not to use existing data in temp dir
	useExistingData = False
	try:
		os.mkdir(tempDataDir)
		os.mkdir(trainDir)
		os.mkdir(validateDir)
		os.mkdir(testDir)
	except FileExistsError: #Temp directory already exists
		#First determine how many subjects exist to give data to user
		numExistTrain = len(os.listdir(trainDir))
		numExistValidate = len(os.listdir(validateDir))
		numExistTest = len(os.listdir(testDir))
		totalExist = numExistTest + numExistTrain + numExistValidate
		if(totalExist != 0):
			proportionExistTrain = numExistTrain/totalExist
			proportionExistValidate = numExistValidate/totalExist
			proportionExistTest = numExistTest/totalExist
		else:
			proportionExistTrain = 0
			proportionExistValidate = 0
			proportionExistTest = 0

		print('Temporary directory already exists with following data')
		print("\tTotal number of subjects: " + str(totalExist))
		print('\tNumber of training subjects: ' + str(numExistTrain) + " (" + str(proportionExistTrain) + ")")
		print('\tNumber of validation subjects: ' + str(numExistValidate) + " (" + str(proportionExistValidate) + ")")
		print('\tNumber of test subjects: ' + str(numExistTest) + " (" + str(proportionExistTest) + ")")			
		print()
		print('Remove data and unpack fresh data? (y/n)')
		deleteData = False #Flag to store whether user wants to delete data
		if(yesNoPrompt(args.default, 'n')): #Prompt user for input
			print("Are you sure? THIS WILL DELETE ALL UNPACKED DATA (y/n)") #Confirm that user actually wants to delete existing data
			if(yesNoPrompt(args.default, 'n')):
				deleteData = True
		if(deleteData):
			shutil.rmtree(tempDataDir)
			os.mkdir(tempDataDir)
			os.mkdir(trainDir)
			os.mkdir(validateDir)
			os.mkdir(testDir)
		else:
			print('Use Existing data? (y/n)')
			if(yesNoPrompt(args.default, 'y')): #Using existing data, no need to unpack new data
				useExistingData = True
				print("Using existing data. Ignoring train and validate proportions provided and using existing distribution.")
			else: #Cannot unpack or use existing, exit program
				raise FileExistsError('Cannot operate on non-empty temp directory. Clean directory or select a new directory.')
	print()
	if(not useExistingData): #Need to unpack new data
		#Unzip data for all subjects, write to temp directory
		print('Unpacking subject data into temporary directory: ' + tempDataDir)
		print('Splitting data into training, validation, and test sets')
		#Randomize order of subjects to randomize test, training, and validations ets
		random.shuffle(subjectDirs)
		#Init Progress bar
		pbar = createProgressBar(maxVal=len(subjectDirs))
		pbar.start()
		for i, subject in pbar(enumerate(subjectDirs)):
			if(i < numTrainSubjects): #Write to training data folder
				with tarfile.open(subject, 'r:*') as f:
					f.extractall(trainDir)
			elif(i < numTrainSubjects + numValidateSubjects): #Validation folder
				with tarfile.open(subject, 'r:*') as f:
					f.extractall(validateDir)
			else: #Test folder
				with tarfile.open(subject, 'r:*') as f:
					f.extractall(testDir)
			pbar.update(i)
		pbar.finish()
		print("Number of training subjects: " + str(numTrainSubjects))
		print('Number of validation subjects: ' + str(numValidateSubjects))
		print('Number of test subjects: ' + str(numTestSubjects))
		print()	
		print('Data initialization successful!')



class DataPreProcessor(Sequence):
	#Class attributes
	
	# Constructor
	# Initializes the pre=processor by unzipping all of the user data to a temp directory
	# and building an index of valid frames and the respective metadata
	# pathToData - Path to the dir containing the zipped subject data
	# pathTemp, - Path to directory in which to create temp dir
	# batchSize - 
	# trainProportion - Proportion of data to make training data
	# validationProportion - Proportion of data to make validation data
	# Note: trainProportion + validateProportion must be <= 1. Remaining percentage
	# frameIndex, metadat, sampledFrames each contain three keys: train, validate, test
	# each key represents a dataset
	# frameIndex - stores sets that contain the filepaths for all valid frames within dataset
	# metadata - stores dictionaries whose keys are the filepaths for valid frame
	# 			each dictionary contains 5 keys: face, leftEye, rightEye, faceGrid, and label
	#			each of these keys refers to a dictionary containing the necessary metadata to describe feature
	# sampledFrames - stores sets that described the data that has already been sampled in the current epoch
	def __init__(self, pathToData, pathTemp, batchSize, dataset, args, debug = False, loadAllData = False):
		self.args = args #Stores all command line arguments

		self.loadedData = False

		self.debug = debug
		if(not(dataset in ['test', 'validate', 'train'])):
			raise ValueError("Invalid dataset. Dataset can only be test, train, or validate.")
		#Creating variables containing paths to data dirs
		self.tempDataDir = pathTemp + '/temp/' + dataset

		#Stores the number of subjects
		self.numSubjects =  len(os.listdir(self.tempDataDir))

		#Build index of metadata for each frame of dataset
		#Stores the paths to all valid frames
		#Note: path is relative to working dir, not training data dir
		if(loadAllData):
			print('Building index, collecting metadata, and loading images for ' + dataset + ' dataset')
		else:
			print('Building index and collecting metadata for ' + dataset + ' dataset')
		if(loadAllData):
			self.frameIndex, self.metadata, self.frames = self.indexData(self.tempDataDir, loadAllData)
			self.loadedData = True
		else:
			self.frameIndex, self.metadata = self.indexData(self.tempDataDir, loadAllData)

		#Get Number of frames
		self.numFrames = len(self.frameIndex)

		print()
		print("Number of " + dataset + " frames: " + str(self.numFrames))
		print()
		print('Data Pre-processor initialization successful for ' + dataset + ' dataset!')
		print()

		#Shuffle order of frame index to randomize batches
		random.shuffle(self.frameIndex)

		#Initializing other variables
		self.batchSize = batchSize

	# cleanup
	# Should be called at the time of destroying the Preprocessor object
	# Deletes the temporary directory from the filesystem
	def cleanup(self):
		print('Removing temp directory...')
		shutil.rmtree(self.tempDataDir)

	# indexData
	# Builds an index of the data for a dataset and a dictionary containing the metadata
	# The index is stored in a set, and contains the filepaths for each valid frame in the dataset
	# The metadata dictionary has the filepaths for each frame as keys.
	# Each frame in the dictionary is another dictionary containing
	#  5 keys corresponding to features for that frame
	# The value for the key is a dictionary of the critical metadata for that key
	# The keys and metadata include:
	# 'face' : X, Y, H, W
	# 'leftEye' : X, Y, H, W
	# 'rightEye' : X, Y, H, W
	# 'faceGrid' : X, Y, H, W
	# 'label' : XCam, YCam
	# XCam and YCam are the locations of the target relative to the camera
	# Arguments:
	# path - Path to the directory containing all of the subject dirs for a dataset
	# Returns:
	# frameIndex - A set containing the filepaths to all valid frames in the dataset
	# metadata = A dictionary containing the metadata for each frame
	def indexData(self, path, loadAllData):
		#Getting unzipped subject dirs
		subjectDirs = os.listdir(path=path)

		#Declare index lists and metadata dictionary to return
		if(loadAllData):
			frames = []
		frameIndex = []
		metadata = {}
		pbar = createProgressBar()
		frameNum = 0
		for subject in pbar(subjectDirs):
			subjectPath = path + "/" + subject
			#Stores the name of the frame files in the frames dir
			frameNames = self.getFramesJSON(subjectPath)
			#Collecting metadata about face, eyes, facegrid, labels
			face = self.getFaceJSON(subjectPath)
			leftEye, rightEye = self.getEyesJSON(subjectPath)
			faceGrid = self.getFaceGridJSON(subjectPath)
			dotInfo = self.getDotJSON(subjectPath)

			#Iterate over frames for the current subject
			for i, (frame, fv, lv, rv, fgv) in enumerate(zip(frameNames,
								face['IsValid'],
								leftEye['IsValid'],
								rightEye['IsValid'],
								faceGrid['IsValid'])):
				#Check if cur frame is valid
				if(fv*lv*rv*fgv == 1):
					#Generate path for frame
					framePath = subjectPath + "/frames/" + frame
					#Write file path to index
					if(not loadAllData):
						frameIndex.append(framePath)
						metadata[framePath] = {
							'face' : {'X' : face['X'][i], 'Y': face['Y'][i], 'W' : face['W'][i], 'H'  : face['H'][i]},
							'leftEye' : {'X' : leftEye['X'][i], 'Y': leftEye['Y'][i], 'W' : leftEye['W'][i], 'H'  : leftEye['H'][i]},
							'rightEye' : {'X' : rightEye['X'][i], 'Y': rightEye['Y'][i], 'W' : rightEye['W'][i], 'H'  : rightEye['H'][i]},
							'faceGrid' : {'X' : faceGrid['X'][i], 'Y': faceGrid['Y'][i], 'W' : faceGrid['W'][i], 'H'  : faceGrid['H'][i]},
							'label': {'XCam' : dotInfo['XCam'][i], 'YCam' : dotInfo['YCam'][i]}
						}
					else:
						frameIndex.append(frameNum)
						with open(framePath, 'rb') as f:
							frames.append(np.fromstring(f.read(), dtype=np.uint8))
						metadata[frameNum] = {
							'face' : {'X' : face['X'][i], 'Y': face['Y'][i], 'W' : face['W'][i], 'H'  : face['H'][i]},
							'leftEye' : {'X' : leftEye['X'][i], 'Y': leftEye['Y'][i], 'W' : leftEye['W'][i], 'H'  : leftEye['H'][i]},
							'rightEye' : {'X' : rightEye['X'][i], 'Y': rightEye['Y'][i], 'W' : rightEye['W'][i], 'H'  : rightEye['H'][i]},
							'faceGrid' : {'X' : faceGrid['X'][i], 'Y': faceGrid['Y'][i], 'W' : faceGrid['W'][i], 'H'  : faceGrid['H'][i]},
							'label': {'XCam' : dotInfo['XCam'][i], 'YCam' : dotInfo['YCam'][i]}
						}
					#Build the dictionary containing the metadata for a frame
					
					frameNum += 1
		if(loadAllData):
			return frameIndex, metadata, frames
		else:
			return frameIndex, metadata

	# getFramesJSON
	# Loads frames.json to a dictionary
	# this file contains the names fo the frames in the directory
	# Arguments:
	# subjectPath - Path to unzipped root directory of the subject
	# Files that are read in:
	# - frames.json
	# Returns dictionary objects containing the ingested JSON object
	def getFramesJSON(self, subjectPath):
		with open(subjectPath + '/frames.json') as f:
			frames = json.load(f)
		return frames

	# getFaceJSON
	# Collects data about face for a specific subject from the json files
	# Arguments:
	# subjectPath - Path to unzipped root directory of the subject
	# Files that are read in:
	# - appleFace.json
	# Returns dictionary objects containing the ingested JSON object
	def getFaceJSON(self, subjectPath):
		with open(subjectPath + '/appleFace.json') as f:
			faceMeta = json.load(f)
		return faceMeta

	# getEyes
	# Collects data about eyes (left & right) for a specific subject from the json files
	# Arguments:
	# subjectPath - Path to unzipped root directory of the subject
	# Files that are read in:
	# - appleLeftEye.json
	# - appleRightEye.json
	# Returns 2 dictionary objects containing the ingested JSON objects
	def getEyesJSON(self, subjectPath):
		with open(subjectPath + '/appleLeftEye.json') as f:
			leftEyeMeta = json.load(f)
		with open(subjectPath + '/appleRightEye.json') as f:
			rightEyeMeta = json.load(f)
		return (leftEyeMeta, rightEyeMeta)

	# getFaceGrid
	# Collects data about the facegrid for a specific subject from the json files
	# Arguments:
	# subjectPath - Path to unzipped root directory of the subject
	# Files that are read in:
	# - faceGrid.json
	# Returns dictionary object containing the ingested JSON object
	def getFaceGridJSON(self, subjectPath):
		with open(subjectPath + '/faceGrid.json') as f:
			faceGridMeta = json.load(f)
		return faceGridMeta

	# getDotJSON
	# Collects data about dot (target subject is looking at) for a specific subject 
	# from the json files
	# Arguments:
	# subjectPath - Path to unzipped root directory of the subject
	# Files that are read in:
	# - dotInfo.json
	# Returns dictionary object containing the ingested JSON object
	def getDotJSON(self, subjectPath):
		with open(subjectPath + '/dotInfo.json') as f:
			dotMeta = json.load(f)
		return dotMeta

	# getImage
	# Reads in image from file
	# imagePath - path to image to retrieve
	# Returns numpy array containing the image
	def getImage(self, imagePath):
		if(self.loadedData):
			return cv2.imdecode(self.frames[imagePath], -1)
		else:
			return 	cv2.imread(imagePath)

	# getInputImages
	# Creates the properly formatted (cropped and scaled) images of the
	# face, left eye, and right eye
	# Arguments:
	# imagePath - List of the paths of the images to retrieve
	# Returns 4D 3 NumPy arrays containing the images (image, x, y, channel)
	def getInputImages(self, imagePaths):
		#Desired size of images after processing
		desiredImageSize = 224

		#Creating numpy arrays to store images
		faceImages = np.zeros((len(imagePaths), desiredImageSize, desiredImageSize, 3))
		leftEyeImages =  np.zeros((len(imagePaths), desiredImageSize, desiredImageSize, 3)) 
		rightEyeImages =  np.zeros((len(imagePaths), desiredImageSize, desiredImageSize, 3))
		
		#Iterate over all imagePaths to retrieve images
		for i, frame in enumerate(imagePaths):
			#Reading in frame from file
			image = self.getImage(frame)

			#Crop image of face from original frame
			xFace = int(self.metadata[frame]['face']['X'])
			yFace = int(self.metadata[frame]['face']['Y'])
			wFace = int(self.metadata[frame]['face']['W'])
			hFace = int(self.metadata[frame]['face']['H'])


			#Crop image of left eye
			#JSON file specifies position eye relative to face
			#Therefore, we must transform to make coordinates
			#Relative to picture by adding coordinates of face
			xLeft = int(self.metadata[frame]['leftEye']['X']) + xFace
			yLeft = int(self.metadata[frame]['leftEye']['Y']) + yFace
			wLeft = int(self.metadata[frame]['leftEye']['W'])
			hLeft = int(self.metadata[frame]['leftEye']['H'])

			#Right Eye
			xRight = int(self.metadata[frame]['rightEye']['X']) + xFace
			yRight = int(self.metadata[frame]['rightEye']['Y']) + yFace
			wRight = int(self.metadata[frame]['rightEye']['W'])
			hRight = int(self.metadata[frame]['rightEye']['H'])
			#Bound checking - ensure x & y are >= 0
			if(xFace < 0):
				wFace = wFace + xFace
				xFace = 0
			if(yFace < 0):
				hFace = hFace + yFace
				yFace = 0
			if(xLeft < 0):
				wLeft = wLeft + xLeft
				xLeft = 0
			if(yLeft < 0):
				hLeft = hLeft + yLeft
				yLeft = 0
			if(xRight < 0):
				wRight = wRight + xRight
				xRight = 0
			if(yRight < 0):
				hRight = hRight + yRight
				yRight = 0


			#Retrieve cropped images
			faceImage = imageUtils.crop(image, xFace, yFace, wFace, hFace)
			leftEyeImage = imageUtils.crop(image, xLeft, yLeft, wLeft, hLeft)
			rightEyeImage = imageUtils.crop(image, xRight, yRight, wRight, hRight)

			#Resize images to 224x224 to pass to neural network
			faceImage = imageUtils.resize(faceImage, desiredImageSize)
			leftEyeImage = imageUtils.resize(leftEyeImage, desiredImageSize)
			rightEyeImage = imageUtils.resize(rightEyeImage, desiredImageSize)

			#Writing process images to np array
			faceImages[i] = faceImage
			leftEyeImages[i] = leftEyeImage
			rightEyeImages[i] = rightEyeImage

		#Noramlize all data to scale 0-1
		faceImages = imageUtils.normalize(faceImages, 255)
		leftEyeImages = imageUtils.normalize(leftEyeImages, 255)
		rightEyeImages = imageUtils.normalize(rightEyeImages, 255)

		return faceImages, leftEyeImages, rightEyeImages

	# getFaceGrids
	# Extract the faceGrid information from JSON to numpy array
	# Arguments:
	# imagePath - List of the paths of the images to retrieve
	# Returns a 2D NumPy array containing the faceGrid (framesx625)
	def getFaceGrids(self, imagePaths):
		#Size of the facegrid output
		faceGridSize = 625

		faceGrids = np.zeros((len(imagePaths), faceGridSize))
		for frameNum, frame in enumerate(imagePaths):
			#Retrieve necessary values
			x =  self.metadata[frame]['faceGrid']['X']
			y =  self.metadata[frame]['faceGrid']['Y']
			w =  self.metadata[frame]['faceGrid']['W']
			h =  self.metadata[frame]['faceGrid']['H']

			#Create 5x5 array of zeros
			faceGrid = np.zeros((25, 25))

			#Write 1 in the FaceGrid for location of face
			#Subtracting 1 in range because facegrid is 1 indexed
			xBound = x-1+w
			yBound = y-1+h

			if(x < 0): #Flooring x & y to zero
				x = 0 
			if(y < 0):
				y = 0
			if(xBound > 25): #Capping maximum value of x & y to 25
				xBound = 25
			if(yBound > 25):
				yBound = 25

			for i in range(x-1,xBound):
				for j in range(y-1,yBound):
					faceGrid[j][i] = 1
			#Reshapre facegird from 25x25 to 625x1
			faceGrid = np.reshape(faceGrid, faceGridSize)
			faceGrids[frameNum] = faceGrid

		return faceGrids

	# getLabels
	# Extract the x and y location of the gaze relative to the camera Frame of Reference
	# Arguments:
	# imagePaths - List of the paths of the images to retrieve
	# Returns a (framesx2) numpy array containing the x and y location of the targets relative to camera
	def getLabels(self, imagePaths):
		labels = np.zeros((len(imagePaths), 2))
		for i, frame in enumerate(imagePaths):
			labels[i] = np.array([self.metadata[frame]['label']['XCam'],
									self.metadata[frame]['label']['YCam']])
		return labels

	# generateBatch
	# Generates a batch of data to pass to ML model
	# The batch contains batchSize number of frames
	# Frames are randomly selected from entire dataset
	# Arguments:
	# batchSize - Number of frames to put in the output batch
	# Returns:
	# Dictionary containing the following keys
	# - face: Numpy array containing batch of data for face (batchSize, 224, 224, 3)
	# - leftEye: Numpy array containing batch of data for left eye (batchSize, 224, 224, 3)
	# - rightEye: Numpy array containing batch of data for right eye (batchSize, 224, 224, 3)
	# - faceGrid: Numpy array containing batch of data for facegrid (batchSize, 625)
	# Numpy array containing label batch (batchSize, 2). 
	# 	The labels are the x & y location of gaze relative to camera.
	# Numpy array containing metadata for batch (batchSize, 2)
	# 	Metadata describes the subject and frame number for each image 
	def __getitem__(self, index):
		startIndex = index*self.batchSize
		try: #Full size batch
			framesToRetrieve = self.frameIndex[startIndex : startIndex + self.batchSize]
		except IndexError: #Retrieve small batch at the end of the array
			framesToRetrieve = self.frameIndex[startIndex:]

		faceBatch, leftEyeBatch, rightEyeBatch = self.getInputImages(framesToRetrieve)
		faceGridBatch = self.getFaceGrids(framesToRetrieve)
		labelsBatch = self.getLabels(framesToRetrieve)
		if(not self.debug):
			return {
						'input_3' : faceBatch, 
						'input_1' : leftEyeBatch, 
						'input_2' : rightEyeBatch, 
						'input_4' : faceGridBatch
					}, labelsBatch#, metaBatch
		else:
			metaBatch = np.array(framesToRetrieve)
			return {
						'input_3' : faceBatch, 
						'input_1' : leftEyeBatch, 
						'input_2' : rightEyeBatch, 
						'input_4' : faceGridBatch
					}, labelsBatch, metaBatch



	# __len__
	# Returns the number of batches in an epoch
	# If the number of frames is not divisible by the batchSize, num batches is rounded up
	# Last batch is smaller than batchSize
	def __len__(self):
		return math.ceil(self.numFrames/self.batchSize)

	def on_epoch_begin(self):
		if(loadedData):
			random.shuffle(self.loadedDataIndex)
		else:
	 		random.shuffle(self.frameIndex)

	


# pp = DataPreProcessor('data/zip', 0.8, 0.15)
# input()
# inputs, labels, meta = pp.generateBatch(50, 'train')
# pp.displayBatch(inputs, labels, meta)
# inputs, labels, meta = pp.generateBatch(10, 'validate')
# pp.displayBatch(inputs, labels, meta)


# pp.cleanup()


