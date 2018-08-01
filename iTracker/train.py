if __name__ == '__main__':
	#Run pre-train config
	try:
		import config
		print("Running config from config.py")
		config.run_config()
	except ImportError:
		print("Unable to load config.py - Executing program with no pre-runtime configuration")

#General imports
import os, shutil
import math, json
from time import time


if __name__ == '__main__':
	#TODO Determine how to use learning rate multipliers
	#TODO Determine how to use grouping in convolutional layer
	#TODO Determine how to create LRN layer

	#Keras imports
	from keras.models import Model, load_model
	from keras.optimizers import SGD
	from keras.utils.training_utils import multi_gpu_model
	from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler #Import callbacks for training

	#Tensorflow device
	import tensorflow as tf
	
	#Custom imports
	from uiUtils import yesNoPrompt #UI prompts
	from customCallbacks import Logger #Logger callback for logging training progress
	import processData 	#Custom datset processor
	import iTrackerModel #Machine learning model import


	############################ LOAD DATA AND HYPERPARAMETERS ############################
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

	#Load Data paths
	pathToData = dataPathJSON['pathToData']
	pathTemp = dataPathJSON['pathToTempDir']
	pathLogging = dataPathJSON['pathLogging']

	#Load machine parameters
	machineParams = paramJSON['machineParameters']
	numWorkers = machineParams['numberofWorkers'] #Number of workers to spawn in parallel
	numGPU = machineParams['numberofGPUS'] #Number of GPUs to use
	useMultiGPU = (numGPU > 1) #Set flag determining number of GPus to use



	#Confirm ML hyperparameters
	print("----------------------------")
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
	print('Number of workers: ' + str(numWorkers))
	print('Number of GPUs:  ' + str(numGPU))
	print("----------------------------")
	print('Train with these parameters? (y/n)')
	if(not yesNoPrompt()): #Delete directory
		raise Exception('Incorrect training parameters. Modify ml_param.json')

	#Load Machine parameters

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

	##################################### IMPORT MODEL ####################################
	#Define Stochastic Gradient descent optimizer
	def getSGDOptimizer():
		return SGD(lr=lrSchedule['0'], momentum=momentum, decay=decay)
	if(not loadModel): #Build and compile ML model, training model from scratch
		if(useMultiGPU):
			with tf.device("/cpu:0"):
				#This is the model to be saved
				iTrackerModel = iTrackerModel.initializeModel() #Retrieve iTracker Model
			iTrackerModelMultiGPU = multi_gpu_model(iTrackerModel, numGPU) 
			print("Using " + str(numGPU) + " GPUs")
		else:
			iTrackerModel = iTrackerModel.initializeModel() #Retrieve iTracker Model
			print("Using 1 GPU")


		#Set initial values
		initialEpoch = 0
		#Compile model
		iTrackerModel.compile(getSGDOptimizer(), loss=['mse'])

	else: #Loading model from file

		#Set initial values
		initialEpoch = previousTrainingState['epoch']
		print("Loading model from file")
		print("Previous training ended at: ")
		print("Epoch: " + str(previousTrainingState['epoch']))
		print("Learning Rate: " + str(previousTrainingState['learningRate']))
		print("Training Loss:  " + str(previousTrainingState['trainLoss']))
		print("Validation Loss:  " + str(previousTrainingState['validateLoss']))
		if(useMultiGPU):
			with tf.device("/cpu:0"):
				#This is the model to be saved
				iTrackerModel = load_model(modelPath) #Retrieve iTracker Model
			iTrackerModelMultiGPU = multi_gpu_model(iTrackerModel, numGPU) 
			#Compile model
			iTrackerModel.compile(getSGDOptimizer(), loss=['mse'])
			print("Using " + str(numGPU) + " GPUs")
		else:
			iTrackerModel = load_model(modelPath)
	#Define functions to retrieve the correct model depending on mutli gpu or not
	def trainModel(): return iTrackerModelMultiGPU if useMultiGPU else iTrackerModel
	def saveModel(): return iTrackerModel  
	################################### DEFINE CALLBACKS ##################################
	logPeriod = 1 #Frequency at which to log data

	#logger callback - Writes the current training state to file to load
	loggerFilepath = pathLogging + '/checkpoints/'
	loggerCallback = Logger(
		saveModel(),
		loggerFilepath,
		hyperParamsJSON,
		period = logPeriod,
		)

	#Tensorboard
	tensorboardFilepath = pathLogging + '/tensorboard'
	tensorboard = TensorBoard(
		log_dir = tensorboardFilepath + '/{}'.format(time()), 
		histogram_freq = 0, 
		write_graph = True, 
		write_images = True)

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

	#Learning Rate Scheduler
	learningRateScheduler = LearningRateScheduler(lrScheduler)
	
	#Adding all callbacks to list to pass to fit generator
	callbacks = [tensorboard, learningRateScheduler, loggerCallback]


	#################################### TRAINING MODEL ###################################
	print("")
	print("Beginning Training...")
	trainModel().fit_generator(
			ppTrain, 
			epochs = numEpochs, 
			validation_data = ppValidate, 
			callbacks = callbacks,
			initial_epoch = initialEpoch,
			use_multiprocessing = False,
			workers = numWorkers
		) #Only use multiprocessing if we are not using multiple GPUs

	#Evaluate model here
	testLoss = iTrackerModel.evaluate_generator(
		ppTest,
		steps = math.ceil(len(ppTest))
	)

	print("Saving trained model to " + pathLogging + "/finalModel.h5")
	saveModel().save(pathLogging + "/finalModel.h5")

	print()
	print("FINISHED MODEL TRAINING AND EVALUATION")
	print("Final test loss: " + str(testLoss))
	print('\nCleanup unpacked data? (y/n)')
	if(yesNoPrompt()):
		ppTrain.cleanup() #Call one on, but deletes temp dir, so deletes tenp data for all datasets

