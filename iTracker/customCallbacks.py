from keras.callbacks import Callback
import keras.backend as K
import json

class Logger(Callback):

	def __init__(self, model, filepath,  hyperparams, period=1):
		super(Logger, self).__init__() 
		self.model = model # Store model to store
		self.logFilepath = filepath + 'iTracker-log-{epoch:02d}-{val_loss:.2f}.json' #Generate filepath for logging
		self.checkpointFilepath = filepath + 'iTracker-checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5' #Generate filepath for checkpoint
		self.period = period #Logging period
		self.epochs_since_last_save = 0
		self.hyperparams = hyperparams 


	def on_epoch_end(self, epoch, logs=None):
		logs = logs or {} #Creating a dictionary object from logs, if empty, logs becomes an empty dict

		self.epochs_since_last_save += 1
		if self.epochs_since_last_save >= self.period: #Need to log
			self.epochs_since_last_save = 0
			print("Generating checkpoint and logs")

			#### Generating log file ####
			#Current train state
			trainState = {
            	"trainLoss" : logs['loss'],
            	"validateLoss" : logs['val_loss'],
            	"epoch" : epoch,
            	"learningRate" : float(K.get_value(self.model.optimizer.lr))
			}
			#Adding hyperparameters to log file
			outputDict = {
            	'trainState' : trainState,
            	'trainingHyperparameters' : self.hyperparams
			}
			#Populate variables in log filepath
			logFilepath = self.logFilepath.format(epoch=epoch + 1, **logs) 
			with open(logFilepath, 'w') as f: 
				json.dump(outputDict, f) #Write to file

			#### Generate checkpoint ####
			#Populate variables in checkpoint filepath
			checkpointFilepath = self.checkpointFilepath.format(epoch=epoch+1, **logs)
			self.model.save(checkpointFilepath) #Save Model
 