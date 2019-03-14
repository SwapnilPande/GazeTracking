if __name__ == '__main__':
        import argparse #Argument parsing

        #Retrieve command line options
        parser = argparse.ArgumentParser()
        parser.add_argument("execution_name", help="Name to identify the train execution - used to name the log files")
        parser.add_argument("-d", "--default", help="Use default options when configuring execution", action='store_true')
        args =  parser.parse_args()
        use_dp = False

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
from time import time, strftime

if __name__ == '__main__':
        #TODO Determine how to use learning rate multipliers
        #TODO Determine how to use grouping in convolutional layer

        #Keras imports
        from keras.models import Model, load_model
        from keras.optimizers import SGD
        from keras.utils.training_utils import multi_gpu_model
        from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler #Import callbacks for training

        #from privacy.analysis.rdp_accountant import compute_rdp
        #from privacy.analysis.rdp_accountant import get_privacy_spent
        #from privacy.optimizers.dp_optimizer import DPGradientDescentOptimizer
        #from privacy.optimizers.gaussian_query import GaussianAverageQuery
        #Tensorflow device
        import tensorflow as tf

        #Custom imports
        from utils.uiUtils import yesNoPrompt #UI prompts
        from utils.customCallbacks import Logger #Logger callback for logging training progress
        from utils import GDataProcessor, DataProcessor,MTCNNDataProcessor, CaliDataProcessor,CaliDataProcessor_LessMemoryTest, FullDataProcessor #Custom datset processor
        import iTrackerModel # Machine learning model import

        class ModelMGPU(Model):
                def __init__(self, ser_model, gpus):
                        pmodel = multi_gpu_model(ser_model, gpus)
                        self.__dict__.update(pmodel.__dict__)
                        self._smodel = ser_model

                def __getattribute__(self, attrname):
                        '''Override load and save methods to be used from the serial-model. The
                        serial-model holds references to the weights in the multi-gpu model.
                        '''
                        # return Model.__getattribute__(self, attrname)
                        if 'load' in attrname or 'save' in attrname:
                                return getattr(self._smodel, attrname)
                        
                        return super(ModelMGPU, self).__getattribute__(attrname)

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
        print(loadModel)
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
        pathLogging = dataPathJSON['pathLogging'] + "/" + args.execution_name + '_' + strftime('%d-%m-%Y_%H-%M')


        #Load machine parameters
        machineParams = paramJSON['machineParameters']
        numWorkers = machineParams['numberofWorkers'] #Number of workers to spawn in parallel
        queueSize = machineParams['queueSize'] #Max size of queue of images
        numGPU = machineParams['numberofGPUS'] #Number of GPUs to use
        loadTrainInMemory = machineParams['loadTrainIntoMemory'] #Whether or not to load training data in memory
        loadValidateInMemory = machineParams['loadValidateIntoMemory'] #Whether or not to load validate data in memory
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
        print('Queue Size: ' + str(queueSize))
        print('Number of GPUs:  ' + str(numGPU))
        print("----------------------------")
        print('Train with these parameters? (y/n)')
        if(not yesNoPrompt(args.default, 'y')): #Delete directory
                raise Exception('Incorrect training parameters. Modify ml_param.json')

        #Load Machine parameters

        DataProcessor.initializeData(pathToData, pathTemp, trainSetProportion, validateSetProportion, args)


        #Initialize Data pre-processor here
        print('loading training data')
        #CaliTrain = CaliDataProcessor.DataPreProcessor(pathTemp, trainBatchSize, 'train', args, loadAllData = loadValidateInMemory)
        #CaliValidate = CaliDataProcessor.DataPreProcessor(pathTemp, trainBatchSize, 'validate', args, loadAllData = loadValidateInMemory)
        #ppTrain = MTCNNDataProcessor.DataPreProcessor(pathTemp, trainBatchSize, 'train', args, loadAllData = loadTrainInMemory)
        ppValidate = MTCNNDataProcessor.DataPreProcessor(pathTemp, validateBatchSize, 'validate', args, loadAllData = loadValidateInMemory)
        
        
        
        #FullTrain = FullDataProcessor.DataPreProcessor(pathTemp, trainBatchSize, 'train', args, loadAllData = loadTrainInMemory,duplicate =100)
        #FullValidate = FullDataProcessor.DataPreProcessor(pathTemp, validateBatchSize, 'validate', args, loadAllData = loadValidateInMemory,duplicate=10)
        
        ppTest = MTCNNDataProcessor.DataPreProcessor(pathTemp, testBatchSize, 'test', args)
        
        #GppTest =  GDataProcessor.DataPreProcessor(pathTemp, testBatchSize, 'test', args)
        
        #Initialize Logging Dir here

        #Will raise prompt if new log directory exists already and contains checkpoints
        if(os.path.isdir(pathLogging)): #Check if log path already exists
                if(os.path.isfile(pathLogging + "/finalModel.h5") or
                        (os.listdir(pathLogging + '/checkpoints'))): #Check if log directory is non-empty
                        print("Logging directory is non-empty and contains the final model or checkpoints from a previous execution.")
                        print("Remove data? (y/n)")
                        if(yesNoPrompt(args.default, 'n')): #Empty logging directory
                                shutil.rmtree(pathLogging)
                                os.mkdir(pathLogging)
                        else:
                                raise FileExistsError('Clean logging directory or select a new directory')

                        #Attempt to create checkpoint & tensorboard directories, do not raise error if directories exist
                        try:
                                os.mkdir(pathLogging + '/checkpoints')
                        except OSError as e:
                                if e.errno != errno.EEXIST:
                                        raise 
                        try:
                                os.mkdir(pathLogging + '/tensorboard')
                        except OSError as e:
                                if e.errno != errno.EEXIST:
                                        raise
        else: #Logging directory does not exist
                print("Creating logging directory")
                os.mkdir(pathLogging)
                os.mkdir(pathLogging + '/checkpoints')
                os.mkdir(pathLogging + '/tensorboard')


        print("")

        ##################################### IMPORT MODEL ####################################
        #Define Stochastic Gradient descent optimizer
        def getSGDOptimizer():
                if use_dp:
                        dp_average_query = GaussianAverageQuery(
                                1.0,
                                1.0*1.1,
                                256)
                        optimizer = DPGradientDescentOptimizer(
                                dp_average_query,
                                256,
                                learning_rate=0.001,
                                unroll_microbatches=True)
                        return optimizer
                else:
                        
                        return SGD(lr=lrSchedule['0'], momentum=momentum, decay=decay)

        
        if(not loadModel): #Build and compile ML model, training model from scratch
                if(useMultiGPU):
                        with tf.device("/cpu:0"):
                                #This is the model to be saved
                                iTrackerModelOriginal = iTrackerModel.initializeModel_lambda() #Retrieve iTracker Model
                        iTrackerModel =ModelMGPU(iTrackerModelOriginal, numGPU) 
                        print("Using " + str(numGPU) + " GPUs")
                else:
                        iTrackerModel = iTrackerModel.initializeModel_lambda() #Retrieve iTracker Model
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
                                iTrackerModelOriginal = load_model(modelPath) #Retrieve iTracker Model
                        iTrackerModel = multi_gpu_model(iTrackerModelOriginal, numGPU) 
                        #Compile model
                        iTrackerModel.compile(getSGDOptimizer(), loss=['mse'])
                        print("Using " + str(numGPU) + " GPUs")
                else:
                        iTrackerModel = load_model(modelPath)
        #Define functions to retrieve the correct model depending on mutli gpu or not
        def trainModel(): return iTrackerModel 
        def saveModel(): return iTrackerModelOriginal if useMultiGPU else iTrackerModel

        ################################### DEFINE CALLBACKS ##################################
        logPeriod = 1 #Frequency at which to log data

        #Model Checkpoint callback
        checkpointFilepath = pathLogging + '/checkpoints/' +'iTracker-checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5'
        checkpointCallback = ModelCheckpoint(
                checkpointFilepath,
                monitor = 'val_loss',
                period = logPeriod
                )

        #logger callback - Writes the current training state to file to load
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
        callbacks = [checkpointCallback, tensorboard, learningRateScheduler, loggerCallback]


        #################################### TRAINING MODEL ###################################
        print("")
        print("Beginning Training...")

        '''
        trainModel().fit_generator(
                        FullTrain, 
                        epochs = numEpochs, 
                        validation_data = FullValidate, 
                        callbacks = callbacks,
                        initial_epoch = initialEpoch,
                        workers = numWorkers,
                        max_queue_size = queueSize,
                        shuffle=True
                )

        '''
        trainModel().fit_generator(
                        ppValidate, 
                        epochs = numEpochs, 
                        validation_data = ppValidate, 
                        callbacks = callbacks,
                        initial_epoch = initialEpoch,
                        workers = numWorkers,
                        max_queue_size = queueSize,
                        shuffle=True
                )

        
        '''
        for i in range(200):
                
                start_epoch = 1000+i
                stop_epoch = start_epoch+1
                trainModel().fit_generator(
                        ppTrain, 
                        epochs = stop_epoch,
                        steps_per_epoch = 1000,
                        validation_data = CaliValidate, 
                        callbacks = callbacks,
                        initial_epoch = start_epoch,
                        workers = numWorkers,
                        max_queue_size = queueSize,
                        shuffle=True
                )
                start_epoch = 2000+5*i
                stop_epoch = start_epoch+5
                trainModel().fit_generator(
                        CaliTrain, 
                        epochs = stop_epoch, 
                        validation_data = CaliValidate, 
                        callbacks = callbacks,
                        initial_epoch = start_epoch,
                        workers = numWorkers,
                        max_queue_size = queueSize,
                        shuffle=True
                )
                start_epoch = 3000+1*i
                stop_epoch = start_epoch+1
                trainModel().fit_generator(
                        CaliTrain, 
                        epochs = stop_epoch, 
                        validation_data = ppValidate, 
                        callbacks = callbacks,
                        initial_epoch = start_epoch,
                        workers = numWorkers,
                        max_queue_size = queueSize,
                        shuffle=True
                )
        
        '''
        #Evaluate model here
        testLoss = iTrackerModel.evaluate_generator(
                ppTest,
                steps = math.ceil(len(ppTest))
        )

        print("Saving trained model to " + pathLogging + "/final_model.h5")
        saveModel().save(pathLogging + "/final_model.h5")

        print()
        print("FINISHED MODEL TRAINING AND EVALUATION")
        print("Final test loss: " + str(testLoss))
        print('\nCleanup unpacked data? (y/n)')
        if(yesNoPrompt(args.default, 'n')):
                print('\nAre you sure? THIS WILL DELETE ALL UNPACKED DATA (y/n)')
                if(yesNoPrompt(args.default, 'n')):
                        ppTrain.cleanup() #Call one on, but deletes temp dir, so deletes tenp data for all datasets
        print("Exiting train.py")

