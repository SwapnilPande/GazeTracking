import socketio
import LoggingNamespace
import JSON
from keras.models import Model, load_model
import 1

import logging
logging.getLogger('socketIO-client').setLevel(logging.DEBUG)
logging.basicConfig()

#Path to the trained model
modelPath = ""
model = load_model(modelPath) 

#Parameters for connecting to server sending OpenPose data
ipAddress = 'localhost'
port = 8000
dataNamespace = ""
sio = socketio.ASyncServer() #Initialize SocketIO server

#Callback function definitions for Socket.io connection events
@sio.on('connect', namespace = dataNamespace)
def connect(sid, environ):
	print("Connected to server, connection id" + str(sid))

@sio.on('disconnect', namespace = dataNamespace)
def disconnect():
	print("Disconnected from server")

@sio.on('reconnect', namespace = dataNamespace)
def reconnect():
	print("Reconnected to server")

@sio.on('data', namespace = dataNamespace)
def on_dataReceived(data): #TODO fix datatype
	data = json.load(data)

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
def preprocessData(metadata):
	face, leftEye, rightEye = getInputImages(metadata)
	faceGrid = getFaceGrid(metadata)

	return {
				'input_3' : face, 
				'input_1' : leftEye, 
				'input_2' : rightEye, 
				'input_4' : faceGrid
				}






# getInputImages
# Creates the properly formatted (cropped and scaled) images of the
# face, left eye, and right eye
# Arguments:
# imagePath - List of the paths of the images to retrieve
# dataset - String describing the dataset that the iamges come from
# 			3 possible values: 'train', 'validate', 'test' 
# Returns 4D 3 NumPy arrays containing the images (image, x, y, channel)
def getInputImages(metadata):
	#Desired size of images after processing
	desiredImageSize = 224

	#Creating numpy arrays to store images
	faceImage = np.zeros((desiredImageSize, desiredImageSize, 3))
	leftEyeImage =  np.zeros((desiredImageSize, desiredImageSize, 3)) 
	rightEyeImage =  np.zeros((desiredImageSize, desiredImageSize, 3))
	
	#Reading in frame from file
	image = getImage(frame) #TODO Retrieve image here

	#Crop image of face from original frame
	xFace = int(metadata['face']['X'])
	yFace = int(metadata['face']['Y'])
	wFace = int(metadata['face']['W'])
	hFace = int(metadata['face']['H'])

	#Crop image of left eye
	#JSON file specifies position eye relative to face
	#Therefore, we must transform to make coordinates
	#Relative to picture by adding coordinates of face
	xLeft = int(metadata['leftEye']['X']) + xFace
	yLeft = int(metadata['leftEye']['Y']) + yFace
	wLeft = int(metadata['leftEye']['W'])
	hLeft = int(metadata['leftEye']['H'])

	#Right Eye
	xRight = int(metadata['rightEye']['X']) + xFace
	yRight = int(metadata['rightEye']['Y']) + yFace
	wRight = int(metadata['rightEye']['W'])
	hRight = int(metadata['rightEye']['H'])

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
	faceImages = imageUtils.normalize(faceImage, 255)
	leftEyeImages = imageUtils.normalize(leftEyeImage, 255)
	rightEyeImages = imageUtils.normalize(rightEyeImage, 255)

	return faceImage, leftEyeImage, rightEyeImage

# getFaceGrids
# Extract the faceGrid information from JSON to numpy array
# Arguments:
# imagePath - List of the paths of the images to retrieve
# dataset - String describing the dataset that the iamges come from
# 			3 possible values: 'train', 'validate', 'test' 	
# Returns a 2D NumPy array containing the faceGrid (framesx625)
def getFaceGrid(metadata):
	#TODO implement retrieving facegrid
	return faceGrids






