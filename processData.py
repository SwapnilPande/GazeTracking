import json
import numpy as np
import cv2
import random
import os
import shutil
import tarfile





class DataPreProcessor:
	#Class attributes
	
	# Constructor
	# Initializes the pre=processor by unzipping all of the user data to a temp directory
	# and building an index of valid frames and the respective metadata
	# pathToData - Path to the dir containing the zipped subject data
	# trainProportion - Proportion of data to make training data
	# validationProportion - Proportion of data to make validation data
	# Note: trainProportion + validateProportion must be <= 1. Remaining percentage
	# is used as training data 
	def __init__(self, pathToData, trainProportion, validateProportion):
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
		self.tempDataDir = 'data/temp'
		self.trainDir = self.tempDataDir + '/train'
		self.validateDir = self.tempDataDir + '/validate'
		self.testDir = self.tempDataDir + '/test'
		try:
			os.mkdir(self.tempDataDir)
			os.mkdir(self.trainDir)
			os.mkdir(self.validateDir)
			os.mkdir(self.testDir)
		except FileExistsError: #Temp directory already exists
			print('Temporary directory already exists. Clean directory? (y/n)')
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
				raise FileExistsError('Cannot operate on non-empty temp directory. Clean directory or select a new directory.')
		print()

		#Number of training, validation, and test subjects
		self.numTrainSubjects = round(trainProportion*self.numSubjects)
		self.numValidateSubjects = round(validateProportion*self.numSubjects)
		self.numTestSubjects = self.numSubjects - self.numTrainSubjects - self.numValidateSubjects

		#Unzip data for all subjects, write to temp directory
		print('Unpacking subject data into temporary directory: ./' + self.tempDataDir)
		print('Splitting data into training, validation, and test sets')
		#Randomize order of subjects to randomize test, training, and validations ets
		random.shuffle(subjectDirs)
		for i, subject in enumerate(subjectDirs):
			if(i < self.numTrainSubjects): #Write to training data folder
				with tarfile.open(subject, 'r:*') as f:
					f.extractall(self.trainDir)
			elif(i < self.numTrainSubjects + self.numValidateSubjects): #Validation folder
				with tarfile.open(subject, 'r:*') as f:
					f.extractall(self.validateDir)
			else: #Test folder
				with tarfile.open(subject, 'r:*') as f:
					f.extractall(self.testDir)
		print("Number of training subjects: " + str(self.numTrainSubjects))
		print('Number of validation subjects: ' + str(self.numValidateSubjects))
		print('Number of test subjects: ' + str(self.numTestSubjects))
		print()

		#Build index of metadata for each frame of training, validation, and testing data
		#Stores the paths to all valid frames (training)
		#Note: path is relative to working dir, not training data dir
		print('Building index and collecting metadata for training data')
		self.trainFrameIndex, self.trainMetaData = self.indexData(self.trainDir)
		
		print('Building index and collecting metadata for validation data')
		self.validateFrameIndex, self.validateMetaData = self.indexData(self.validateDir)
		
		print('Building index and collecting metadata for testing data')
		self.testFrameIndex, self.testMetaData = self.indexData(self.testDir) 

		#Get Number of frames
		self.numTrainFrames = len(self.trainFrameIndex)
		self.numValidateFrames = len(self.validateFrameIndex)
		self.numTestFrames = len(self.testFrameIndex)

		print()
		print("Number of training frames: " + str(self.numTrainFrames))
		print('Number of validation frames: ' + str(self.numValidateFrames))
		print('Number of testing frames: ' + str(self.numTestFrames))
		print()
		print('Initialization successful!')

		#Initializing other variables
		#Used to store the frames that have already been samples this epoch
		self.sampledTrainFrames = []
		self.sampledValidateFrames = []

		
	def cleanup(self):
		print('Removing temp directory...')
		shutil.rmtree(self.tempDataDir)


	def indexData(self, path):
		#Getting unzipped subject dirs
		subjectDirs = os.listdir(path=path)

		#Declare index lists and metadata dictionary to return
		frameIndex = []
		metadata = {}
		for subject in subjectDirs:
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
					framePath = subjectPath + "/" + frame
					#Write file path to index
					frameIndex.append(framePath)
					#Build the dictionary containing the metadata for a frame
					metadata[framePath] = {
						'face' : {'X' : face['X'][i], 'Y': face['Y'][i], 'W' : face['W'][i], 'H'  : face['H'][i]},
						'leftEye' : {'X' : leftEye['X'][i], 'Y': leftEye['Y'][i], 'W' : leftEye['W'][i], 'H'  : leftEye['H'][i]},
						'rightEye' : {'X' : rightEye['X'][i], 'Y': rightEye['Y'][i], 'W' : rightEye['W'][i], 'H'  : rightEye['H'][i]},
						'faceGrid' : {'X' : faceGrid['X'][i], 'Y': faceGrid['Y'][i], 'W' : faceGrid['W'][i], 'H'  : faceGrid['H'][i]},
						'label': ['XCam' : dotInfo['XCam'][i], 'YCam' : dotInfo['YCam'][i]]
					}
		return frameIndex, metadata

	# getFramesJOSN
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
	# Subject - Integer containing the number of the subject (name of the subject directory)
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
	# Subject - Integer containing the number of the subject (name of the subject directory)
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
	# Subject - Integer containing the number of the subject (name of the subject directory)
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
	# Subject - Integer containing the number of the subject (name of the subject directory)
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
		return 	cv2.imread(
			self.getSubjectPath(subject) + 
			'/frames/' + 
			self.num2Str(frame, 5) + 
			'.jpg')



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
			image = self.getImage(subject, frame)

			faceMeta = {}
			leftEyeMeta = {}
			rightEyeMeta = {}

			#Retrieve metadata about given frame
			if frame in self.trainMetaData:
				faceMeta = self.trainMetaData[frame]['face']
				leftEyeMeta = self.trainMetaData[frame]['leftEye']
				rightEyeMeta = self.trainMetaData[frame]['rightEye']
			elif frame in self.validateMetaData:
				faceMeta = self.validateMetaData[frame]['face']
				leftEyeMeta = self.validateMetaData[frame]['leftEye']
				rightEyeMeta = self.validateMetaData[frame]['rightEye']
			else:
				faceMeta = self.testMetaData[frame]['face']
				leftEyeMeta = self.testMetaData[frame]['leftEye']
				rightEyeMeta = self.testMetaData[frame]['rightEye']


			#Crop image of face from original frame
			xFace = int(faceMeta['X'])
			yFace = int(faceMeta['Y'])
			wFace = int(faceMeta['W'])
			hFace = int(faceMeta['H'])
			faceImage = self.crop(image, xFace, yFace, wFace, hFace)

			#Crop image of left eye
			#JSON file specifies position eye relative to face
			#Therefore, we must transform to make coordinates
			#Relative to pictuer by adding coordinates of face
			x = int(leftEyeMeta['X']) + xFace
			y = int(leftEyeMeta['Y'])+ yFace
			w = int(leftEyeMeta['W'])
			h = int(leftEyeMeta['H'])
			leftEyeImage = self.crop(image, x, y, w, h)

			#Right Eye
			x = int(rightEyeMeta['X']) + xFace
			y = int(rightEyeMeta['Y']) + yFace
			w = int(rightEyeMeta['W'])
			h = int(rightEyeMeta['H'])
			rightEyeImage = self.crop(image, x, y, w, h)

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
	# Returns a 2D NumPy array containing the faceGrid (framesx625)
	def getFaceGrids(self, imagePaths):
		#Size of the facegrid output
		faceGridSize = 625

		faceGrids = np.zeros((len(imagePaths), faceGridSize))
		for frameNum, frame in enumerate(imagePaths):
			faceGridMeta = {}
			#Retrieve metadata about given frame
			if frame in self.trainMetaData:
				faceGridMeta = self.trainMetaData[frame]['faceGrid']
			elif frame in self.validateMetaData:
				faceGridMeta = self.validateMetaData[frame]['faceGrid']
			else:
				faceGridMeta = self.testMetaData[frame]['faceGrid']

			x = faceGridMeta['X']
			y = faceGridMeta['Y']
			w = faceGridMeta['W']
			h = faceGridMeta['H']

			#Create 5x5 array of zeros
			faceGrid = np.zeros((25, 25))

			#Write 1 in the FaceGrid for location of face
			#Subtracting 1 in range because facegrid is 1 indexed
			for i in range(x-1,x-1+w):
				for j in range(y-1,y-1+h):
					faceGrid[j][i] = 1
			#Reshapre facegird from 25x25 to 625x1
			faceGrid = np.reshape(faceGrid, faceGridSize)
			faceGrids[frameNum] = faceGrid

		return faceGrids

	# getLabels
	# Extract the x and y location of the gaze relative to the camera Frame of Reference
	# Arguments:
	# imagePath - List of the paths of the images to retrieve
	# Returns a (framesx2) numpy array containing the x and y location of the targets relative to camera
	def getLabels(self, imagePaths):
		labels = np.zeros((len(imagePaths), 2))
		for i, frame in enumerate(imagePaths):
			labelMeta = {}
			#Retrieve metadata about given frame
			if frame in self.trainMetaData:
				labelMeta = self.trainMetaData[frame]['label']
			elif frame in self.validateMetaData:
				labelMeta = self.validateMetaData[frame]['label']
			else:
				labelMeta = self.testMetaData[frame]['label']
			labels[i] = np.array([labelMeta['XCam'], labelMeta['YCam']])
		return labels

	# getMaxFrames
	# Determines the maximum number of frames that can be retrieved from a subject
	# Returns value of maxFrames if the number of valid frames exceeds maxFrames
	# Else, returns the number of valid frames
	# Arguments:
	# subject - Integer number of subject for which to determine max frames
	# Returns integer containing max number of frames
	def getMaxFrames(self, subject):
		#Collecting information about subject from info files
		subjectInfo = self.getSubjectInfoJSON(subject)
		numEyeDetections = subjectInfo['NumEyeDetections']
		numFaceDetections = subjectInfo['NumFaceDetections']
		return min((self.maxFrames, numEyeDetections, numFaceDetections))


	# selectRandomFrames
	# Generates a list of length numFrames of random and valid frames 
	# for a given subject
	# Arguments:
	# subject - Integer number of subject for which to determine max frames
	# numFrames - Integer describing number of frames to retrieve
	# 			numFrames must be less than or equal to num valid frames
	# Returns a list of frames to collect
	def selectRandomFrames(self, subject, numFrames):
		#Collecting information about subject from info files
		faceInfo = self.getFaceGridJSON(subject)
		leftEyeInfo, rightEyeInfo = self.getEyesJSON(subject)

		validFrames = []
		#Create list of all indices with valid frames
		for i, (f, l, r) in enumerate(zip(
							faceInfo["IsValid"], 
							leftEyeInfo['IsValid'], 
							rightEyeInfo['IsValid'])):
			if(f*l*r == 1):
				validFrames.append(i)
		#randomize order of frames
		random.shuffle(validFrames)

		#Return only the first numFrames frames
		return validFrames[:numFrames]

	# generateBatch
	# Generates a batch of data to pass to ML model
	# The batch contains batchSize number of frames
	# A random number of frames (2 <= maxFrames < maxFrames)
	# are selected from randomly selected participants
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
	def generateBatch(self):
		#Intializing numpy array for output batches
		#metaBatch containsL:
		# subject number - indeex 0
		# frame number - index 1
		metaBatch = np.zeros((self.batchSize, 2))
		faceBatch = np.zeros((self.batchSize, 224, 224, 3))
		leftEyeBatch = np.zeros((self.batchSize, 224, 224, 3))
		rightEyeBatch = np.zeros((self.batchSize, 224, 224, 3))
		faceGridBatch = np.zeros((self.batchSize, 625))
		labelsBatch = np.zeros((self.batchSize, 2))

		#Keeps count of the number of frames collected in batch
		numSamples = 0
		while(numSamples < self.batchSize):
			#Randomly select a subject
			subject = random.randint(0, self.numSubjects-1)

			#Select random number of frames where 2 <= num <= maxFrames
			numFrames = random.randint(2, self.getMaxFrames(subject))

			#Truncate number of frames if batch size will be exceeded
			if(numSamples + numFrames > self.batchSize):
				numFrames = self.batchSize - numSamples

			#Select numFrames valid frames
			frames = self.selectRandomFrames(subject, numFrames)
			#Retrieve images and facegrids for the selected frames
			face, left, right = self.getInputImages(subject, frames)
			faceGrid = self.getFaceGrids(subject, frames)
			labels = self.getLabels(subject, frames)

			#Iterating over all collected data and adding to batch
			# fr - frame
			# f - face
			# l - left
			# r - right
			# fg = facegrid
			# lb - labels
			for fr, f, l, r, fg, lb in zip(frames, face, left, right, faceGrid, labels):
				metaBatch[numSamples][0] = subject
				metaBatch[numSamples][1] = fr 
				faceBatch[numSamples] = f
				leftEyeBatch[numSamples] = l
				rightEyeBatch[numSamples] = r
				faceGridBatch[numSamples] = fg
				labelsBatch[numSamples] = lb
				numSamples = numSamples + 1
		return {
					'face' : faceBatch, 
					'leftEye' : leftEyeBatch, 
					'rightEye' : rightEyeBatch, 
					'faceGrid' : faceGridBatch
				}, labelsBatch, metaBatch

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
			title += ': Subject ' + str(meta[0]) 
			title += ', Frame ' + str(meta[1]) 

			print(title)

			#Draw facegrid here
			print('FACEGRID')
			for j in range(0,25):
				for k in range(0,25):
					print(int(fg[k+25*j]), end='')
				print('')

			print('LABELS')
			print(lb)
			#Place 3 images side by side to display
			output = np.concatenate((f, l, r), axis = 1)
			#Show images
			cv2.imshow(title, output)

			#Wait for key input
			key = cv2.waitKey(0)
			if(key == 27):
				break
			cv2.destroyAllWindows()
	
# pp = DataPreProcessor(2, 50, 10)
# inputs, labels, meta =  pp.generateBatch()
# pp.displayBatch(inputs, labels, meta)

pp = DataPreProcessor('data/zip', 0.8, 0.15)
input()
pp.cleanup()


