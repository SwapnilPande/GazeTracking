try:
	import config
	config.run_config()
except ImportError:
	print("Unable to load config.py")
	print("Executing program with no pre-runtime configuration")

#TODO Determine how to use learning rate multipliers
#TODO Determine how to use grouping in convolutional layer
#TODO Determine how to create LRN layer
from keras.models import Model, load_model

#Import necessary layers for model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Concatenate, Reshape, ZeroPadding2D

#Import initializers for weights and biases
from keras.initializers import Zeros, RandomNormal

from keras.optimizers import SGD

#Import callbacks for training
from keras.callbacks import ModelCheckpoint

#Custom datset processor
from processData import DataPreProcessor

import math
import os

import json #used to read config file

#Function definitions for defining model
def randNormKernelInitializer():
	return RandomNormal(stddev= 0.01)

# createConvLayer
# Function to simplify the process of creating a convolutional layer for iCapture
# Populates parameters that are common for all convolutional layers in network
#
# INPUTS
# filters - Number of feature layers in the output
# kernel_size - dimension of kernel in pixels - creates square kernel
# stride - Stride taken during convolution
#
# Returns a Conv2D object describing the new layer
def createConvLayer(filters, kernelSize, stride):
	return Conv2D(
		filters,
		kernelSize, 
		strides = stride,
		activation = 'relu',
		use_bias = True,
		kernel_initializer = randNormKernelInitializer(),
		bias_initializer = 'zeros'
		)

# createMaxPool
# Function to simplify the process of creating a MaxPooling layer
# Populates parameters that are common for all maxpool layers in net
# Returns a MaxPooling2D object describing the new layer
def createMaxPool():
	return MaxPooling2D(pool_size = 3, strides = 2)

def createPadding(pad):
	return ZeroPadding2D(padding=pad)


def createFullyConnected(units):
	return Dense(
		units,
	 	activation = 'relu',
	 	use_bias = True,
	 	kernel_initializer = randNormKernelInitializer(),
		bias_initializer = 'zeros'
	 	)

################### Begin Execution ####################
#Define ML parameters
with open('ml_param.json') as f:
	#Boolean to store whether the training should start from scratch
	#Or if prexisting model should be loaded
	paramJSON = json.load(f)
	loadModel = paramJSON['loadPrexistingModel']
	if(not loadModel):
		learningRate = paramJSON['learningRate']
		momentum = paramJSON['momentum']
		decay = paramJSON['decay']

		numEpochs = paramJSON['numEpochs']
		trainBatchSize = paramJSON['trainBatchSize']
		validateBatchSize = paramJSON['validateBatchSize']
		testBatchSize = paramJSON['testBatchSize']

		trainSetProportion = paramJSON['trainSetProportion']
		validateSetProportion = paramJSON['validateSetProportion']
	else:
		modelPath = paramJSON['prexistingModelPath']
	pathToData = paramJSON['pathToData']
	pathTemp = paramJSON['pathToTempDir']
	pathLogging = paramJSON['pathLogging']

print()
#Confirm ML Parameters
print('Loaded following parameters from ml_param.json.')
print('Learning Rate: ' + str(learningRate))
print('Momentum: ' + str(momentum))
print('Decay: ' + str(decay))
print()
print("Number of Epochs: " + str(numEpochs))
print("Training Batch Size: " + str(trainBatchSize))
print("Validation Batch Size: " + str(validateBatchSize))
print("Test Batch Size: " + str(testBatchSize))
print()
print("Training Set Proportion: " + str(trainSetProportion))
print('Validation Set Proportion: ' + str(validateSetProportion))
print("Testing Set Proportion: " + str(1 - trainSetProportion - validateSetProportion))
print()
print('Path to data: ' + pathToData)
print('Path to create temp directory: ' + pathTemp)
print('Path to store logs: ' + pathLogging)
print()
print('Train with these parameters? (y/n)')
response = input()
while(response != 'y' and response != 'n'):
	print("Enter only y or n:")
	response = input()
if(response == 'n'): #Delete directory
	raise Exception('Incorrect training parameters. Modify ml_param.json')

#Initialize Data pre-processor here
pp = DataPreProcessor(pathToData, pathTemp, trainSetProportion, validateSetProportion)


#Training model from scratch:
if(not loadModel):
	#Defining input here
	leftEyeInput = Input(shape=(224,224,3,))
	rightEyeInput = Input(shape=(224,224,3,))
	faceInput = Input(shape=(224,224,3,))
	faceGridInput = Input(shape=(625,))

	#Define convolutional layers for left and right eye inputs
	convE1 = createConvLayer(96, 11, 4)
	maxPoolE1 = createMaxPool()
	paddingE1 = createPadding(2)
	convE2 = createConvLayer(256, 5, 1)
	maxPoolE2 = createMaxPool()
	paddingE2 = createPadding(1)
	convE3 = createConvLayer(384, 3, 1)
	convE4 = createConvLayer(64, 1, 1)

	#Define convolutional layers for face input
	convF1 = createConvLayer(96, 11, 4)
	maxPoolF1 = createMaxPool()
	paddingF1 = createPadding(2)
	convF2 = createConvLayer(256, 5, 1)
	maxPoolF2 = createMaxPool()
	paddingF2 = createPadding(1)
	convF3 = createConvLayer(384, 3, 1)
	convF4 = createConvLayer(64, 1, 1)

	#Define fully connected layer for left & right eye concatenation
	fullyConnectedE1 = createFullyConnected(128)

	#Define fully connected layers for face
	fullyConnectedF1 = createFullyConnected(128)
	fullyConnectedF2 = createFullyConnected(64)

	#Define fully connected layers for face grid
	fullyConnectedFG1 = createFullyConnected(256)
	fullyConnectedFG2 = createFullyConnected(128)

	#Define fully connected layers for eyes & face & face grid
	fullyConnected1 = createFullyConnected(128)
	fullyConnected2 = createFullyConnected(2)


	#Defining dataflow through layers
	#Left Eye
	leftDataConvE1 = convE1(leftEyeInput)
	leftDataMaxPoolE1 = maxPoolE1(leftDataConvE1)
	leftDataPaddingE1 = paddingE1(leftDataMaxPoolE1)
	leftDataConvE2 = convE2(leftDataPaddingE1)
	leftDataMaxPoolE2 = maxPoolE2(leftDataConvE2)
	leftDataPaddingE2 = paddingE2(leftDataMaxPoolE2)
	leftDataConvE3 = convE3(leftDataPaddingE2)
	leftDataConvE4 = convE4(leftDataConvE3)
	#Reshape data to feed into fully connected layer
	leftEyeFinal = Reshape((9216,))(leftDataConvE4)

	#Right Eye
	rightDataConvE1 = convE1(rightEyeInput)
	rightDataMaxPoolE1 = maxPoolE1(rightDataConvE1)
	rightDataPaddingE1 = paddingE1(rightDataMaxPoolE1)
	rightDataConvE2 = convE2(rightDataPaddingE1)
	rightDataMaxPoolE2 = maxPoolE2(rightDataConvE2)
	rightDataPaddingE2 = paddingE2(rightDataMaxPoolE2)
	rightDataConvE3 = convE3(rightDataPaddingE2)
	rightDataConvE4 = convE4(rightDataConvE3)
	#Reshape data to feed into fully connected layer
	rightEyeFinal = Reshape((9216,))(rightDataConvE4)

	#Combining left & right eye
	dataLRMerge = Concatenate(axis=1)([leftEyeFinal, rightEyeFinal])
	dataFullyConnectedE1 = fullyConnectedE1(dataLRMerge)

	#Face
	dataConvF1 = convF1(faceInput)
	dataMaxPoolF1 = maxPoolF1(dataConvF1)
	dataPaddingF1 = paddingF1(dataMaxPoolF1)
	dataConvF2 = convF2(dataPaddingF1)
	dataMaxPoolF2 = maxPoolF2(dataConvF2)
	dataPaddingF2 = paddingF2(dataMaxPoolF2)
	dataConvF3 = convF3(dataPaddingF2)
	dataConvF4 = convF4(dataConvF3)
	#Reshape data to feed into fully connected layer
	faceFinal = Reshape((9216,))(dataConvF4)
	dataFullyConnectedF1 = fullyConnectedF1(faceFinal)
	dataFullyConnectedF2 = fullyConnectedF2(dataFullyConnectedF1)


	#Face grid
	dataFullyConnectedFG1 = fullyConnectedFG1(faceGridInput)
	dataFullyConnectedFG2 = fullyConnectedFG2(dataFullyConnectedFG1)

	#Combining Eyes & Face & Face Grid
	finalMerge = Concatenate(axis=1)([dataFullyConnectedE1, dataFullyConnectedF2, dataFullyConnectedFG2])
	dataFullyConnected1 = fullyConnected1(finalMerge)
	finalOutput = fullyConnected2(dataFullyConnected1)

	#Initializing the model
	iTrackerModel = Model(inputs = [leftEyeInput, rightEyeInput, faceInput, faceGridInput], outputs = finalOutput)
	#Define Stochastic Gradient descent optimizer
	def getSGDOptimizer():
		return SGD(lr=learningRate, momentum=momentum, decay=decay)
	#Compile model
	iTrackerModel.compile(getSGDOptimizer(), loss=['mean_squared_error'], metrics=['accuracy'])


	#Define the callback to checkpoint the model 
	os.mkdir(pathLogging + '/checkpoints')
	checkpointFilepath = pathLogging + '/checkpoints/' +'iTracker-checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5'
	checkpointCallback = ModelCheckpoint(
		checkpointFilepath,
		monitor='val_loss',
		period=10
		)
else: #Loading model from file
	iTrackerModel = load_model(modelPath)

iTrackerModel.fit_generator(
		pp.generateBatch(trainBatchSize, 'train'), 
		epochs = numEpochs, 
		steps_per_epoch = math.ceil(pp.numTrainFrames/trainBatchSize), 
		validation_data = pp.generateBatch(validateBatchSize, 'validate'), 
		validation_steps = math.ceil(pp.numValidateFrames/validateBatchSize),
		callbacks = [checkpointCallback]
	)


#Evaluate model here
testLoss = iTrackerModel.evaluate_generator(
	pp.generateBatch(testBatchSize, 'test'),
	steps = math.ceil(pp.numTestFrames/testBatchSize)
)

iTrackerModel.save(pathLogging + "/finalModel.h5")

print()
print("FINISHED MODEL TRAINING AND EVALUATION")
print("Final test loss: " + str(testLoss))
pp.cleanup()

