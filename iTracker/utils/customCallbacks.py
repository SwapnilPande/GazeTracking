from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
import json

class Logger(Callback):

	def __init__(self, filepath,  hyperparams, period=1):
		super(Logger, self).__init__() 
		self.filepath = filepath #Save filepath for logging
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
			filepath = self.filepath.format(epoch=epoch + 1, **logs) 
			with open(filepath, 'w') as f: 
				json.dump(outputDict, f) #Write to file
 
