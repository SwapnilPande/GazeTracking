import json
import numpy as np
import cv2

# num2Str
# Creates a string containng zero padded integer
# Arguments:
# number - integer to convert to string
# length - fixed length of string >= length of integer
# Returns zero padded string
def num2Str(number, length):
	return format(number, '0' + str(length) + 'd')

# getFaceJSON
# Collects data about face for a specific frame from the json files
# Arguments:
# Subject - Integer containing the number of the subject (name of the subject directory)
# Files that are read in:
# - appleFace.json
# Returns dictionary objects containing the ingested JSON object
def getFaceJSON(subject):
	with open(num2Str(subject, 5) + '/appleFace.json') as f:
		faceMeta = json.load(f)
	return faceMeta

# getEyes
# Collects data about eyes (left & right) for a specific frame from the json files
# Arguments:
# Subject - Integer containing the number of the subject (name of the subject directory)
# Files that are read in:
# - appleLeftEye.json
# - appleRightEye.json
# Returns 2 dictionary objects containing the ingested JSON objects
def getEyesJSON(subject):
	with open(num2Str(subject, 5) + '/appleLeftEye.json') as f:
		leftEyeMeta = json.load(f)
	with open(num2Str(subject, 5) + '/appleRightEye.json') as f:
		rightEyeMeta = json.load(f)
	return (leftEyeMeta, rightEyeMeta)

# getFaceGrid
# Collects data about the facegrid for a specific frame from the json files
# Arguments:
# Subject - Integer containing the number of the subject (name of the subject directory)
# Files that are read in:
# - faceGrid.json
# Returns dictionary object containing the ingested JSON object
def getFaceGridJSON(subject):
	with open(num2Str(subject, 5) + '/faceGrid.json') as f:
		faceGridMeta = json.load(f)
	return faceGridMeta

# getDotJSON
# Collects data about dot (target subject is looking at) for a specific frame 
# from the json files
# Arguments:
# Subject - Integer containing the number of the subject (name of the subject directory)
# Files that are read in:
# - dotInfo.json
# Returns dictionary object containing the ingested JSON object
def getDotJSON(subject):
	with open(num2Str(subject, 5) + '/dotInfo.json') as f:
		dotMeta = json.load(f)
	return dotMeta

# getImage
# Reads in image from file
# subjeect - Integer containing subject number from which to collect the frame
# frame - Integer containing the frame number from which to extract eyes
# Returns numpy array containing the image
def getImage(subject, frame):
	return 	cv2.imread(
		num2Str(subject, 5) + 
		'/frames/' + 
		num2Str(frame, 5) + 
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
def crop(image, x, y, w, h):
	return image[y:y+h, x:x+w]

# resize
# Resizes an image based on the provided parameters
# Not destructive, does not modify original file
# Arguments:
# image - NumPy array containing the image
# imageSizePx - Final size of image in pixels. This is both height and weidth
# Returns numpy array containing the resized image
def resize(image, imageSizePx):
	return cv2.resize(image, (imageSizePx, imageSizePx))
# normalize
# Rescales all of the data to a scale of 0-1
# Arguments:
# scale - current max value of data
# returns NumPy array scaled from 0-1
def normalize(image, maxVal):
	return np.divide(image, maxVal)


# getInputArrays
# Creates the properly formatted (cropped and scaled) images of the
# face, left eye, and right eye
# Arguments:
# subjeect - Integer containing subject number from which to collect the frame
# frame - Integer containing the frame number from which to extract eyes
# Returns 3 NumPy arrays containing the images
def getInputImages(subject, frame):
	#Reading in data about face and eyes from JSON files
	faceJSON = getFaceJSON(subject)
	leftEyeJSON, rightEyeJSON = getEyesJSON(subject)

	#Reading in frame from file
	image = getImage(subject, frame)


	#Crop image of face from original frame
	x = int(faceJSON['X'][frame])
	y = int(faceJSON['Y'][frame])
	w = int(faceJSON['W'][frame])
	h = int(faceJSON['H'][frame])
	faceImage = crop(image, x, y, w, h)

	#Crop image of left eye from cropped face image
	x = int(leftEyeJSON['X'][frame])
	y = int(leftEyeJSON['Y'][frame])
	w = int(leftEyeJSON['W'][frame])
	h = int(leftEyeJSON['H'][frame])
	leftEyeImage = crop(faceImage, x, y, w, h)

	#Right Eye
	x = int(rightEyeJSON['X'][frame])
	y = int(rightEyeJSON['Y'][frame])
	w = int(rightEyeJSON['W'][frame])
	h = int(rightEyeJSON['H'][frame])
	rightEyeImage = crop(faceImage, x, y, w, h)

	#Resize images to 224x224 to pass to neural network
	faceImage = resize(faceImage, 224)
	leftEyeImage = resize(leftEyeImage, 224)
	rightEyeImage = resize(rightEyeImage, 224)

	#Noramlize all data to scale 0-1
	faceImage = normalize(faceImage, 224)
	leftEyeImage = normalize(leftEyeImage, 224)
	rightEyeImage = normalize(rightEyeImage, 224)

	return faceImage, leftEyeImage, rightEyeImage

# getFaceGrid
# Extract the faceGrid information from JSON to numpy array
# Arguments:
# subjeect - Integer containing subject number from which to collect the frame
# frame - Integer containing the frame number from which to extract eyes
# Returns a 625x1 NumPy array containing the faceGrid
def getFaceGrid(subject, frame):
	#Reading in data facegrid data
	faceGridJSON = getFaceGridJSON(subject)

	x = faceGridJSON['X'][frame]
	y = faceGridJSON['Y'][frame]
	w = faceGridJSON['W'][frame]
	h = faceGridJSON['H'][frame]

	#Create 5x5 array of zeros
	faceGrid = np.zeros((25, 25))

	#Write 1 in the FaceGrid for location of face
	for i in range(x,x+w):
		for j in range(y,y+h):
			faceGrid[j][i] = 1

	#Reshapre facegird from 25x25 to 625x1
	faceGrid = np.reshape(faceGrid, 625)

	return faceGrid

# getLabels
# Extract the x and y location of the gaze relative to the camera Frame of Reference
# Arguments:
# subjeect - Integer containing subject number from which to collect the frame
# frame - Integer containing the frame number from which to extract eyes
# Returns a 2x1 array containing the x and y location of the target relative to camera
def getLabels(subject, frame):
	dotJSON = getDotJSON(subject)
	return np.array([dotJSON['XCam'][frame], dotJSON['YCam'][frame]])
	






	


