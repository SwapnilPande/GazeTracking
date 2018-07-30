from uiUtils import yesNoPrompt
#Custom datset processor
import processData
import uiUtils
import json
import cv2
import numpy as np

with open('ml_param.json') as f: #Open paramter file
	paramJSON = json.load(f) #Load JSON as dict

dataPathJSON = paramJSON['dataPaths'] # Extract datapath information
pathToData = dataPathJSON['pathToData']
pathTemp = dataPathJSON['pathToTempDir']

trainSetProportion = paramJSON['trainingHyperparameters']['trainSetProportion']
validateSetProportion = paramJSON['trainingHyperparameters']['validateSetProportion']

#Initialize Data pre-processor here
processData.initializeData(pathToData, pathTemp, trainSetProportion, validateSetProportion)

#Prompt user to select dataset
print("Which dataset would you like to visualize?")
dataset = uiUtils.listOptionsPrompt(['train','validate','test', 'exit'])

while(dataset != 'exit'):
	pp = processData.DataPreProcessor(pathToData, pathTemp, 1, dataset, debug = True)
	numImages = len(pp) #Get the total number of images in the dataset
	for i in range(0, numImages):
		input, labels, meta = pp.__getitem__(i)#Title String
		title =  'Image #' + str(i) 
		title += ', ' + str(meta[0]).replace(pathTemp,'')
		print(title)

		print('LABELS')
		print(labels[0]) 
		fgDisplay = cv2.resize(np.reshape(input['input_4'], (25,25)), (224,224))
		fgDisplay = np.stack((fgDisplay, fgDisplay, fgDisplay), axis = 2)
		
		#Place 3 images side by side to display
		output = np.concatenate((input['input_3'][0], fgDisplay, input['input_1'][0], input['input_2'][0]), axis = 1)
		#Show images
		cv2.imshow(title, output)


		#Wait for key input
		key = cv2.waitKey(0)
		cv2.destroyAllWindows()
		if(key == 27):
			break
	print()
	print("Which dataset?")
	dataset = uiUtils.listOptionsPrompt(['train','validate','test', 'exit'])

