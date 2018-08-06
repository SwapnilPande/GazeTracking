# ARGS
import argparse #Argument parsing
#Retrieve command line options
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--default", help="Use default options when configuring execution", action='store_true')
parser.add_argument("data_directory", help="Path to directory containing the unzipped data. Directory should contain temp directory containing 'train', 'validate', and 'test' directories.")
requiredNamed = parser.add_argument_group("required named arguments")
requiredNamed.add_argument('-m', '--model-path', required=True, help = 'Path to trained model')
args = parser.parse_args()

from utils.uiUtils import datasetUI #Importing UI object
from utils.predictUtils import Predictor
from utils import DataProcessor

import json
import numpy as np
import cv2
import time, datetime


#Initializing UI Object
print("Initializing UI")
ui = datasetUI()

predictor = Predictor(args.model_path)
ppTest = DataProcessor.DataPreProcessor(args.data_directory, 1, 'validate', args, loadAllData = False)

for inputs, label in ppTest:
	prediction = predictor.predict(inputs)
	ui.updateUI(prediction, label[0])


