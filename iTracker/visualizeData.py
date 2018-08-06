# ARGS
import argparse #Argument parsing
#Retrieve command line options
parser = argparse.ArgumentParser()
parser.add_argument("data_directory", help="Directory containing the unzipped data. Directory should contain 'train', 'validate', and 'test' directories.")
parser.add_argument("-d", "--default", help="Use default options when configuring execution", action='store_true')
args = parser.parse_args()


from utils.uiUtils import yesNoPrompt
from utils import uiUtils
#Custom datset processor
from utils import DataProcessor
import json
import cv2
import numpy as np

#Prompt user to select dataset
print("Which dataset would you like to visualize?")
dataset = uiUtils.listOptionsPrompt(['train','validate','test', 'exit'])

while(dataset != 'exit'):
	pp = DataProcessor.DataPreProcessor(args.data_directory, 1, dataset, args, debug = True)
	for i, (inputs,labels, meta) in enumerate(pp):
		title =  'Image #' + str(i) 
		title += ', ' + str(meta[0]).replace(args.data_directory,'')
		print(title)

		print('LABELS')
		print(labels[0]) 
		fgDisplay = cv2.resize(np.reshape(inputs['input_4'], (25,25)), (224,224))
		fgDisplay = np.stack((fgDisplay, fgDisplay, fgDisplay), axis = 2)
		
		#Place 3 images side by side to display
		output = np.concatenate((inputs['input_3'][0], fgDisplay, inputs['input_1'][0], inputs['input_2'][0]), axis = 1)
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

