from keras.models import Model, load_model

class Predictor:

	def __init__(self, modelFilePath):
		self.modelFilePath = modelFilePath

		print()
		print("Loading trained iTracker Model")
		self.model = load_model(modelFilePath) 

	def predict(self, input):
		input1Shape = input['input_1'].shape
		input2Shape = input['input_2'].shape
		input3Shape = input['input_3'].shape
		input4Shape = input['input_4'].shape

		#Validate Input
		if(input1Shape[-3:] == (224,224,3) and
			input2Shape[-3:] == (224,224,3) and
			input3Shape[-3:] == (224,224,3) and
			input4Shape[-1:] == (625,) and
			input1Shape[0] == input2Shape[0] 
			== input3Shape[0] == input4Shape[0]):
			return self.model.predict(input)
		else:
			print("Invalid Input")
			return (None,None)
