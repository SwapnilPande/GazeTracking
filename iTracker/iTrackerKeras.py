if __name__ == '__main__':
	#Run Config
	try:
		import config
		config.run_config()
	except ImportError:
		print("Unable to load config.py - Executing program with no pre-runtime configuration")


from customCallbacks import Logger

import os, shutil, math, json
from time import time

from uiUtils import yesNoPrompt
#Custom datset processor
import processData


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
# lrScheduler
# Function for the Learning Rate Scheduler to output the LR based on the provided input
#  If dictionary contains entry for current epoch, function returns associated learning rate from dict
# Else, will return current learning rate
# Arguments:
# epoch - The current epoch number, 0-indexed
# learningRate - Current learning Rate
# Returns learning rate as floating point number
def lrScheduler(epoch, learningRate):
	if(str(epoch) in lrSchedule): #Update learning rate
		print("Setting learning rate: Epoch #" + str(epoch) + ', Learning Rate = ' + str(lrSchedule[str(epoch)]))
		return lrSchedule[str(epoch)]
	return learningRate #No need to update learning rate, return current learning rate

######################### Begin Execution ###############################
if __name__ == '__main__':
	#TODO Determine how to use learning rate multipliers
	#TODO Determine how to use grouping in convolutional layer
	#TODO Determine how to create LRN layer
	from keras.models import Model, load_model
	#Import necessary layers for model
	from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Concatenate, Reshape, ZeroPadding2D
	#Import initializers for weights and biases
	from keras.initializers import Zeros, RandomNormal
	from keras.optimizers import SGD
	from keras.utils.training_utils import multi_gpu_model

	#Import callbacks for training
	from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
	################### LOAD DATA AND HYPERPARAMETERS #######################
	lrSchedule = {} #Dict containing epoch as key, and learning rate as value

	#Define ML parameters
	with open('ml_param.json') as f: #Open paramter file
		paramJSON = json.load(f) #Load JSON as dict

	#hyperParamsJSON stores dict containing training hyperparameters
	#Loaded from ml_params.json if training from scratch
	#Loaded from prexisting log file if training from existing model
	dataPathJSON = paramJSON['dataPaths'] # Extract datapath information
	loadModel = paramJSON['loadPrexistingModel'] #Check if we are loading existing model
	if(not loadModel): #Training from scratch, not loading existing model
		#Load hyperparameters from ml_params.json
		hyperParamsJSON = paramJSON['trainingHyperparameters']
		print()
		print('Loaded following parameters from ml_param.json.')
	else: #Training from existing model
		existingModelPaths = paramJSON['existingModelPaths'] #Retrieve data about existing mdoel
		modelPath = existingModelPaths['prexistingModelPath'] #Retrieve path to existing model
		trainLogFile = existingModelPaths['trainLogFile'] #Retrieve training log file from previous execution
		#Collect training parameters from log files
		print()
		print("Loading training parameters from previous execution log files instead of from ml_params.json")
		print()
		with open(trainLogFile) as f:
			#Boolean to store whether the training should start from scratch
			#Or if prexisting model should be loaded
			logFileJSON = json.load(f)
		#Load hyperparameters from previous execution log file
		hyperParamsJSON = logFileJSON['trainingHyperparameters']
		previousTrainingState = logFileJSON['trainState']
		print("Loaded following parameters from " + trainLogFile)


	#Handle constant or scheduled learning rate
	learningRate = hyperParamsJSON['learningRate']
	try: # Try to treat learningRate as dict
		for epoch in learningRate:
			#Creating lrSchedule dictionary and formatting key strings
			# Convert to int and back to string to delete extra digits
			lrSchedule[str(int(epoch))] = learningRate[epoch] 
	except TypeError: #LearnignRate is single constant value
		lrSchedule = { '0' : learningRate } #Create dictionary with Learning Rate

	#Loading hyperparameters into individual variables
	momentum = hyperParamsJSON['momentum']
	decay = hyperParamsJSON['decay']

	numEpochs = hyperParamsJSON['numEpochs']
	trainBatchSize = hyperParamsJSON['trainBatchSize']
	validateBatchSize = hyperParamsJSON['validateBatchSize']
	testBatchSize = hyperParamsJSON['testBatchSize']

	trainSetProportion = hyperParamsJSON['trainSetProportion']
	validateSetProportion = hyperParamsJSON['validateSetProportion']

	pathToData = dataPathJSON['pathToData']
	pathTemp = dataPathJSON['pathToTempDir']
	pathLogging = dataPathJSON['pathLogging']

	#Confirm ML Parameters
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
	if(not yesNoPrompt()): #Delete directory
		raise Exception('Incorrect training parameters. Modify ml_param.json')

	processData.initializeData(pathToData, pathTemp, trainSetProportion, validateSetProportion)


	#Initialize Data pre-processor here
	ppTrain = processData.DataPreProcessor(pathToData, pathTemp, trainBatchSize, 'train')
	ppValidate = processData.DataPreProcessor(pathToData, pathTemp, validateBatchSize, 'validate')
	ppTest =  processData.DataPreProcessor(pathToData, pathTemp, testBatchSize, 'test')
	#Initialize Logging Dir here
	if(os.path.isfile(pathLogging + "/finalModel.h5") or
		(os.path.isdir(pathLogging + '/checkpoints'))):
		print("Logging directory is non-empty and contains the final model or checkpoints from a previous execution.")
		print("Remove data? (y/n)")
		if(yesNoPrompt()): #Empty logging directory
			shutil.rmtree(pathLogging)
			os.mkdir(pathLogging)
		else:
			raise FileExistsError('Clean logging directory or select a new directory')
		os.mkdir(pathLogging + '/checkpoints')
		os.mkdir(pathLogging + '/tensorboard')
	print("")


	#DEFINING CALLBACKS HERE
	logPeriod = 10 #Frequency at which to log data
	#Checkpoints
	checkpointFilepath = pathLogging + '/checkpoints/' +'iTracker-checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5'
	checkpointCallback = ModelCheckpoint(
		checkpointFilepath,
		monitor = 'val_loss',
		period = logPeriod
		)

	#logger
	loggerFilepath = pathLogging + '/checkpoints/' +'iTracker-log-{epoch:02d}-{val_loss:.2f}.json'
	loggerCallback = Logger(
		loggerFilepath,
		hyperParamsJSON,
		period = logPeriod
		)

	#Tensorboard
	tensorboardFilepath = pathLogging + '/tensorboard'
	tensorboard = TensorBoard(
		log_dir = tensorboardFilepath + '/{}'.format(time()), 
		histogram_freq = 0, 
		write_graph = True, 
		write_images = True)

	#Learning Rate Scheduler
	learningRateScheduler = LearningRateScheduler(lrScheduler)
	#Adding all callbacks to list to pass to fit generator
	callbacks = [checkpointCallback, tensorboard, learningRateScheduler, loggerCallback]

	#Training model from scratch:
	if(not loadModel): #Build and compile ML model
		print("Initializing Model")
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

		#Set initial values
		initialEpoch = 0

		#Initializing the model
		iTrackerModel = Model(inputs = [leftEyeInput, rightEyeInput, faceInput, faceGridInput], outputs = finalOutput)
		iTrackerModelMultiGPU = multi_gpu_model(iTrackerModel, gpus=8)

		#Define Stochastic Gradient descent optimizer
		def getSGDOptimizer():
			return SGD(lr=lrSchedule['0'], momentum=momentum, decay=decay)
		#Compile model
		iTrackerModelMultiGPU.compile(getSGDOptimizer(), loss=['mean_squared_error'], metrics=['accuracy'])


	else: #Loading model from file
		#Set initial values
		initialEpoch = previousTrainingState['epoch']
		print("Loading model from file")
		print("Previous training ended at: ")
		print("Epoch: " + str(previousTrainingState['epoch']))
		print("Learning Rate: " + str(previousTrainingState['learningRate']))
		print("Training Accuracy: " + str(previousTrainingState['trainAccuracy']), end= " ")
		print("Training Loss:  " + str(previousTrainingState['trainLoss']))
		print("Validation Accuracy: " + str(previousTrainingState['validateAccuracy']), end= " ")
		print("Validation Loss:  " + str(previousTrainingState['validateLoss']))
		iTrackerModel = load_model(modelPath)

		iTrackerModel.load_weights(modelPath)




	#Training model
	print("")
	print("Beginning Training...")
	iTrackerModelMultiGPU.fit_generator(
			ppTrain, 
			epochs = numEpochs, 
			validation_data = ppValidate, 
			callbacks = callbacks,
			initial_epoch = initialEpoch,
			use_multiprocessing = True,
			workers = 1
		)


	#Evaluate model here
	testLoss = iTrackerModel.evaluate_generator(
		ppTest,
		steps = math.ceil(pp.numTestFrames/testBatchSize)
	)

	iTrackerModel.save(pathLogging + "/finalModel.h5")

	print()
	print("FINISHED MODEL TRAINING AND EVALUATION")
	print("Final test loss: " + str(testLoss))
	pp.cleanup()

