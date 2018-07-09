from keras.callbacks import Callback
import keras.backend as K
import json

class Logger(Callback):

	def __init__(self, filepath, hyperparams, period=1):
		super(Logger, self).__init__()
		self.filepath = filepath
		self.period = period
		self.epochs_since_last_save = 0
		self.hyperparams = hyperparams


	def on_epoch_end(self, epoch, logs=None):
		logs = logs or {}

		self.epochs_since_last_save += 1
		if self.epochs_since_last_save >= self.period:
			self.epochs_since_last_save = 0
			print("Generating checkpoint and logs")
			trainState = {
            	"trainAccuracy" : logs['acc'],
            	"trainLoss" : logs['loss'],
            	"validateAccuracy" : logs['val_acc'],
            	"validateLoss" : logs['val_loss'],
            	"epoch" : epoch,
            	"learningRate" : float(K.get_value(self.model.optimizer.lr))
			}
			outputDict = {
            	'trainState' : trainState,
            	'trainingHyperparameters' : self.hyperparams
			}
			filepath = self.filepath.format(epoch=epoch + 1, **logs)
			with open(filepath, 'w') as f:
				json.dump(outputDict, f)