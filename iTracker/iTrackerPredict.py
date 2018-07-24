from socketIO_client_nexus import SocketIO, LoggingNamespace
import json
from keras.models import Model, load_model
from iTrackerUI import iTrackerUI #Importing UI object
import numpy as np

#Importing execution parameters
with open('predict_param.json') as f: #Open paramter file
	paramJSON = json.load(f) #Load JSON as dict


######################################## UI Definition ########################################
#Initializing UI Object
print("Initializing UI")
#ui = iTrackerUI()

# updateUI
# Calls the update function for pygame ui to redraw cursor at updated gze location
# arguments:
# 	gazePrediction - tuple containing (x,y) estimate of gaze location relative to camera
def updateUI(gazePrediction):
	ui.updateCursor(gazePrediction)


####################### Machine Learning Model & Tensorflow Definitions #######################
#Path to the trained model
print()
print("Loading trained iTracker Model")
modelPath = paramJSON['prexistingModelPath']
model = load_model(modelPath) 


# getFaceGrids
# Extract the faceGrid information from JSON to numpy array
# Arguments:
# 	fgMetadata - dictionary containing metadata about facegrid 	
# Returns a numpy array containing teh facegrids of length 625
def getFaceGrid(fgMetadata):
	#Size of the facegrid output
	faceGridSize = 625

	#Retrieve necessary values
	x =  fgMetadata['X']
	y =  fgMetadata['Y']
	w =  fgMetadata['W']
	h =  fgMetadata['H']

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

	#Reshape facegird from 25x25 to 625x1
	faceGrid = np.reshape(faceGrid, faceGridSize)

	return faceGrid

def predictGaze(data):
	frameData = json.load(data)

	#Extracting numpy arrays from string
	#They are 1D at first and need to be reshaped
	face = np.fromstring(frameData['image']['face'])
	face = face.reshape(face, (224, 224))

	leftEye = np.fromstring(frameData['image']['leftEye'])
	leftEye = leftEye.reshape(leftEye, (224,224))

	rightEye = np.fromstring(frameData['image']['rightEye'])
	rightEye = rightEye.reshape(rightEye, (224,224))

	#Generate facegrid from metadata
	faceGrid = getFaceGrid(frameData['frameInfo']['faceGrid'])

	predictionInput = {
				'input_3' : face, 
				'input_1' : leftEye, 
				'input_2' : rightEye, 
				'input_4' : faceGrid
				}

	prediction = model.predict(predictionInput)
	print(prediction)
	updateUI(prediction)


#################################### SocketIO Definitions #####################################
def on_connect():
    print('Connected')

def on_disconnect():
    print('disconnect')

def on_reconnect():
    print('reconnect')

#Passes receeived frame data to predictGaze function to genreate predicted gaze location
def frameReceived(data):
	print("Received Frame")
	predictGaze(data)

#Parameters for connecting to server sending OpenPose data
ipAddress = paramJSON['dataStreamServerAddress']
port = paramJSON['dataStreamServerPort']

#Initialize SocketIO Server
print()
print("Starting SocketIO Client")
socketIO = SocketIO(ipAddress, port, LoggingNamespace)

#Standard definitions for connection events
socketIO.on('connect', on_connect)
socketIO.on('disconnect', on_disconnect)
socketIO.on('reconnect', on_reconnect)

#Definition for frame received events, called frameReceived function to update UI
socketIO.on('frame', frameReceived)

#Wait for incoming messages
socketIO.wait()




