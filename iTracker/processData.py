import json
import numpy as np
import cv2
import random
import os
import shutil
import tarfile

#Used to display progress bars for file init
from progressbar import ProgressBar 





class DataPreProcessor:
	#Class attributes
	
	# Constructor
	# Initializes the pre=processor by unzipping all of the user data to a temp directory
	# and building an index of valid frames and the respective metadata
	# pathToData - Path to the dir containing the zipped subject data
	# pathTemp, - Path to directory in which to create temp dir
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
	def __init__(self, pathToData, pathTemp, trainProportion, validateProportion):
		self.trainProportion = trainProportion
		self.validateProportion = validateProportion
		self.testProportion = 1 - trainProportion - validateProportion
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
		self.numSubjects = len(subjectDirs)
		print('Found ' + str(self.numSubjects) + " subjects")
		print()

		#Create data to store unzipped subject data
		#Create three subdirectories to split train, validate, and test data
		print('Creating temporary directory to store unzipped data')
		self.tempDataDir = pathTemp + '/temp'
		self.trainDir = self.tempDataDir + '/train'
		self.validateDir = self.tempDataDir + '/validate'
		self.testDir = self.tempDataDir + '/test'

		#Number of training, validation, and test subjects
		self.numTrainSubjects = round(trainProportion*self.numSubjects)
		self.numValidateSubjects = round(validateProportion*self.numSubjects)
		self.numTestSubjects = self.numSubjects - self.numTrainSubjects - self.numValidateSubjects
		#Flag to store whether or not to use existing data in temp dir
		useExistingData = False
		try:
			os.mkdir(self.tempDataDir)
			os.mkdir(self.trainDir)
			os.mkdir(self.validateDir)
			os.mkdir(self.testDir)
		except FileExistsError: #Temp directory already exists
			#First determine how many subjects exist to give data to user
			numExistTrain = len(os.listdir(self.trainDir))
			numExistValidate = len(os.listdir(self.validateDir))
			numExistTest = len(os.listdir(self.testDir))
			totalExist = numExistTest + numExistTrain + numExistValidate
			print('Temporary directory already exists with following data')
			print("\tTotal number of subjects: " + str(totalExist))
			print('\tNumber of training subjects: ' + str(numExistTrain) + " (" + str(numExistTrain/totalExist) + ")")
			print('\tNumber of validation subjects: ' + str(numExistValidate) + "( " + str(numExistValidate/totalExist) + ")")
			print('\tNumber of test subjects: ' + str(numExistTest) + " (" + str(numExistTest/totalExist) + ")")			
			print()
			print('Remove data and unpack fresh data? (y/n)')
			response = input()
			while(response != 'y' and response != 'n'):
				print("Enter only y or n:")
				response = input()
			if(response == 'y'): #Delete directory
				shutil.rmtree(self.tempDataDir)
				os.mkdir(self.tempDataDir)
				os.mkdir(self.trainDir)
				os.mkdir(self.validateDir)
				os.mkdir(self.testDir)
			else:
				print('Use Existing data? (y/n)')
				response = input()
				while(response != 'y' and response != 'n'):
					print("Enter only y or n:")
					response = input()
				if(response == 'y'): #Using existing data, no need to unpack new data
					useExistingData = True
					print("Using existing data. Ignoring train and validate proportions provided and using existing distribution.")
					self.numTrainSubjects = numExistTrain
					self.numValidateSubjects = numExistValidate
					self.numTestSubjects = numExistTest
					self.numSubjects = totalExist
				else: #Cannot unpack or use existing, exit program
					raise FileExistsError('Cannot operate on non-empty temp directory. Clean directory or select a new directory.')
		print()
		if(not useExistingData): #Need to unpack new data
			#Unzip data for all subjects, write to temp directory
			print('Unpacking subject data into temporary directory: ' + self.tempDataDir)
			print('Splitting data into training, validation, and test sets')
			#Randomize order of subjects to randomize test, training, and validations ets
			random.shuffle(subjectDirs)
			#Init Progress bar
			pbar = ProgressBar(maxval=len(subjectDirs))
			pbar.start()
			for i, subject in pbar(enumerate(subjectDirs)):
				if(i < self.numTrainSubjects): #Write to training data folder
					with tarfile.open(subject, 'r:*') as f:
						f.extractall(self.trainDir)
				elif(i < self.numTrainSubjects + self.numValidateSubjects): #Validation folder
					with tarfile.open(subject, 'r:*') as f:
						f.extractall(self.validateDir)
				else: #Test folder
					with tarfile.open(subject, 'r:*') as f:
						f.extractall(self.testDir)
				pbar.update(i)
			pbar.finish()
			print("Number of training subjects: " + str(self.numTrainSubjects))
			print('Number of validation subjects: ' + str(self.numValidateSubjects))
			print('Number of test subjects: ' + str(self.numTestSubjects))
			print()

		#Build index of metadata for each frame of training, validation, and testing data
		#Stores the paths to all valid frames (training)
		#Note: path is relative to working dir, not training data dir
		self.frameIndex = {} #Declaring dictionary of indexes for training, validation, and testing datasets
		self.metadata = {} #Deeclaring dictionary of metadata dictionaries
		print('Building index and collecting metadata for training data')
		self.frameIndex['train'], self.metadata['train'] = self.indexData(self.trainDir)
		
		print('Building index and collecting metadata for validation data')
		self.frameIndex['validate'], self.metadata['validate'] = self.indexData(self.validateDir)
		
		print('Building index and collecting metadata for testing data')
		self.frameIndex['test'], self.metadata['test'] = self.indexData(self.testDir) 

		#Get Number of frames
		self.numTrainFrames = len(self.frameIndex['train'])
		self.numValidateFrames = len(self.frameIndex['validate'])
		self.numTestFrames = len(self.frameIndex['test'])

		print()
		print("Number of training frames: " + str(self.numTrainFrames))
		print('Number of validation frames: ' + str(self.numValidateFrames))
		print('Number of testing frames: ' + str(self.numTestFrames))
		print()
		print('Initialization successful!')

		#Initializing other variables
		#Used to store the frames that have already been samples this epoch
		self.sampledFrames = {
			'train' : set(),
			'validate' : set(),
			'test' : set()
 		}

	# cleanup
	# Should be called at the time of destroying the Preprocessor object
	# Deletes the temporary directory from the filesystem
	def cleanup(self):
		print('Cleanup unpacked data? (y/n)')
		response = input()
		while(response != 'y' and response != 'n'):
			print("Enter only y or n: ")
			response = input()
		if(response == 'y'):
			print('Removing temp directory...')
			shutil.rmtree(self.tempDataDir)
		print("Exiting program")

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
	def indexData(self, path):
		#Getting unzipped subject dirs
		subjectDirs = os.listdir(path=path)

		#Declare index lists and metadata dictionary to return
		frameIndex = []
		metadata = {}
		pbar = ProgressBar()
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
					frameIndex.append(framePath)
					#Build the dictionary containing the metadata for a frame
					metadata[framePath] = {
						'face' : {'X' : face['X'][i], 'Y': face['Y'][i], 'W' : face['W'][i], 'H'  : face['H'][i]},
						'leftEye' : {'X' : leftEye['X'][i], 'Y': leftEye['Y'][i], 'W' : leftEye['W'][i], 'H'  : leftEye['H'][i]},
						'rightEye' : {'X' : rightEye['X'][i], 'Y': rightEye['Y'][i], 'W' : rightEye['W'][i], 'H'  : rightEye['H'][i]},
						'faceGrid' : {'X' : faceGrid['X'][i], 'Y': faceGrid['Y'][i], 'W' : faceGrid['W'][i], 'H'  : faceGrid['H'][i]},
						'label': {'XCam' : dotInfo['XCam'][i], 'YCam' : dotInfo['YCam'][i]}
					}
		return set(frameIndex), metadata

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
		return 	cv2.imread(imagePath)



	# crop
	# Crops based on the provided parameters
	# Not destructive, does not modify original file
	# Arguments:
	# image - NumPy array containing the image
	# x - Horizontal position to start crop from top left corner
	# y - Vertical position to start crop from top left corner
	# w - width of the crop (horizontal)
	# h - height of the crop (vertical)# Returns numpy array describing the cropped image
	def crop(self, image, x, y, w, h):
		return image[y:y+h, x:x+w]

	# resize
	# Resizes an image based on the provided parameters
	# Not destructive, does not modify original file
	# Arguments:
	# image - NumPy array containing the image
	# imageSizePx - Final size of image in pixels. This is both height and weidth
	# Returns numpy array containing the resized image
	def resize(self, image, imageSizePx):
		return cv2.resize(image, (imageSizePx, imageSizePx))
	# normalize
	# Rescales all of the data to a scale of 0-1
	# Arguments:
	# scale - current max value of data
	# returns NumPy array scaled from 0-1
	def normalize(self, image, maxVal):
		return np.divide(image, maxVal)


	# getInputImages
	# Creates the properly formatted (cropped and scaled) images of the
	# face, left eye, and right eye
	# Arguments:
	# imagePath - List of the paths of the images to retrieve
	# dataset - String describing the dataset that the iamges come from
	# 			3 possible values: 'train', 'validate', 'test' 
	# Returns 4D 3 NumPy arrays containing the images (image, x, y, channel)
	def getInputImages(self, imagePaths, dataset):
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
			xFace = int(self.metadata[dataset][frame]['face']['X'])
			yFace = int(self.metadata[dataset][frame]['face']['Y'])
			wFace = int(self.metadata[dataset][frame]['face']['W'])
			hFace = int(self.metadata[dataset][frame]['face']['H'])


			#Crop image of left eye
			#JSON file specifies position eye relative to face
			#Therefore, we must transform to make coordinates
			#Relative to picture by adding coordinates of face
			xLeft = int(self.metadata[dataset][frame]['leftEye']['X']) + xFace
			yLeft = int(self.metadata[dataset][frame]['leftEye']['Y']) + yFace
			wLeft = int(self.metadata[dataset][frame]['leftEye']['W'])
			hLeft = int(self.metadata[dataset][frame]['leftEye']['H'])

			#Right Eye
			xRight = int(self.metadata[dataset][frame]['rightEye']['X']) + xFace
			yRight = int(self.metadata[dataset][frame]['rightEye']['Y']) + yFace
			wRight = int(self.metadata[dataset][frame]['rightEye']['W'])
			hRight = int(self.metadata[dataset][frame]['rightEye']['H'])
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
			faceImage = self.crop(image, xFace, yFace, wFace, hFace)
			leftEyeImage = self.crop(image, xLeft, yLeft, wLeft, hLeft)
			rightEyeImage = self.crop(image, xRight, yRight, wRight, hRight)

			#Resize images to 224x224 to pass to neural network
			faceImage = self.resize(faceImage, desiredImageSize)
			leftEyeImage = self.resize(leftEyeImage, desiredImageSize)
			rightEyeImage = self.resize(rightEyeImage, desiredImageSize)

			#Writing process images to np array
			faceImages[i] = faceImage
			leftEyeImages[i] = leftEyeImage
			rightEyeImages[i] = rightEyeImage

		#Noramlize all data to scale 0-1
		faceImages = self.normalize(faceImages, 255)
		leftEyeImages = self.normalize(leftEyeImages, 255)
		rightEyeImages = self.normalize(rightEyeImages, 255)

		return faceImages, leftEyeImages, rightEyeImages

	# getFaceGrids
	# Extract the faceGrid information from JSON to numpy array
	# Arguments:
	# imagePath - List of the paths of the images to retrieve
	# dataset - String describing the dataset that the iamges come from
	# 			3 possible values: 'train', 'validate', 'test' 	
	# Returns a 2D NumPy array containing the faceGrid (framesx625)
	def getFaceGrids(self, imagePaths,dataset):
		#Size of the facegrid output
		faceGridSize = 625

		faceGrids = np.zeros((len(imagePaths), faceGridSize))
		for frameNum, frame in enumerate(imagePaths):
			#Retrieve necessary values
			x =  self.metadata[dataset][frame]['faceGrid']['X']
			y =  self.metadata[dataset][frame]['faceGrid']['Y']
			w =  self.metadata[dataset][frame]['faceGrid']['W']
			h =  self.metadata[dataset][frame]['faceGrid']['H']

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
	# dataset - String describing the dataset that the iamges come from
	# 			3 possible values: 'train', 'validate', 'test' 
	# Returns a (framesx2) numpy array containing the x and y location of the targets relative to camera
	def getLabels(self, imagePaths, dataset):
		labels = np.zeros((len(imagePaths), 2))
		for i, frame in enumerate(imagePaths):
			labels[i] = np.array([self.metadata[dataset][frame]['label']['XCam'],
									self.metadata[dataset][frame]['label']['YCam']])
		return labels

	# generateBatch
	# Generates a batch of data to pass to ML model
	# The batch contains batchSize number of frames
	# Frames are randomly selected from entire dataset
	# Arguments:
	# batchSize - Number of frames to put in the output batch
	# dataset - String describing the dataset that the iamges come from
	# 			3 possible values: 'train', 'validate', 'test' 
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
	def generateBatch(self, batchSize, dataset):
		while True:		
			#Determine frames that have been unused in this epoch
			#by subtracting the sampledTrainFrames from trainFrameIndex  
			unusedFrames = self.frameIndex[dataset] - self.sampledFrames[dataset]
			self.frameIndex[dataset]
			if(len(unusedFrames) > batchSize):  
				#Collect batchSize number of  random frames
				framesToRetrieve = set(random.sample(unusedFrames, batchSize)) 
				#Mark frames in current batch as used
				self.sampledFrames[dataset] = self.sampledFrames[dataset] | framesToRetrieve
			else: #Not enough unused frames to fill batch. Returning remaining frames
				framesToRetrieve = unusedFrames
				#Clear sampled trained frames since all frames have now been sampled in this epoch
				self.sampledFrames[dataset] = set()

			#Generating batches here
			#Convert set framesToRetrieve to list so that order is preserved for all data
			framesToRetrieve = list(framesToRetrieve)

			#metaBatch = np.array(framesToRetrieve)
			faceBatch, leftEyeBatch, rightEyeBatch = self.getInputImages(framesToRetrieve, dataset)
			faceGridBatch = self.getFaceGrids(framesToRetrieve, dataset)
			labelsBatch = self.getLabels(framesToRetrieve, dataset)

			yield {
						'input_3' : faceBatch, 
						'input_1' : leftEyeBatch, 
						'input_2' : rightEyeBatch, 
						'input_4' : faceGridBatch
					}, labelsBatch#, metaBatch

	# displayBatch
	# Displays the data for each frame in the batch
	# Prints facegrid and labels to console and displays images
	# Arguments:
	# input - Dictionary containing face, left eye, right eye, and facegrid data
	# labels - Numpy array containing the labels for each frame
	# meta - Numpy array containing subject & frame number for each frame
	# Note: Arguments are created by generateBatch() and should be passed
	# without modification
	# Press esc to stop outputting frame, press any key to go to next frame
	def displayBatch(self, input, labels, meta):
		for i, (f, l, r, fg, lb, meta) in enumerate(zip(input['face'],
													input['leftEye'],
													input['rightEye'],
													input['faceGrid'],
													labels,
													meta)):
			#Title String
			title =  'Image #' + str(i) 
			title += ': Subject ' + str(meta)
			print(title)

			print('LABELS')
			print(lb)
			fgDisplay = cv2.resize(np.reshape(fg, (25,25)), (224,224))
			fgDisplay = np.stack((fgDisplay, fgDisplay, fgDisplay), axis = 2)
			
			#Place 3 images side by side to display
			output = np.concatenate((f, fgDisplay, l, r), axis = 1)
			#Show images
			cv2.imshow(title, output)


			#Wait for key input
			key = cv2.waitKey(0)
			if(key == 27):
				break
			cv2.destroyAllWindows()
	


# pp = DataPreProcessor('data/zip', 0.8, 0.15)
# input()
# inputs, labels, meta = pp.generateBatch(50, 'train')
# pp.displayBatch(inputs, labels, meta)
# inputs, labels, meta = pp.generateBatch(10, 'validate')
# pp.displayBatch(inputs, labels, meta)


# pp.cleanup()


