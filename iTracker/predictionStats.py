# ARGS
import argparse #Argument parsing
#Retrieve command line options
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--default", help="Use default options when configuring execution", action='store_true')
parser.add_argument("data_directory", help="Path to directory containing the unzipped data. Directory should contain temp directory containing 'train', 'validate', and 'test' directories.")
requiredNamed = parser.add_argument_group("required named arguments")
requiredNamed.add_argument('-m', '--model-path', required=True, help = 'Path to trained model')
args = parser.parse_args()

from utils.uiUtils import createProgressBar #Importing UI object
from utils.predictUtils import Predictor
from utils import DataProcessor
from utils.dataUtils import getDescriptiveStats

import json
import numpy as np
import cv2
import time, datetime

data = {
    'xError' : [],
    'yError' : []
}

predictor = Predictor(args.model_path)
ppTest = DataProcessor.DataPreProcessor(args.data_directory, 1, 'validate', args, loadAllData = False)


print('Iterating over test dataset to generate predictions')
#Init Progress bar
pbar = createProgressBar(maxVal=len(ppTest))
print(len(ppTest))
pbar.start()
for i in range(len(ppTest)):
    inputs, labels = ppTest[i]
    predictions = predictor.predict(inputs)
    for label, prediction in zip(labels, predictions):
         data['xError'].append(prediction[0] - label[0])
         data['yError'].append(prediction[1] - label[1])
    pbar.update(i)

print(getDescriptiveStats(data))
    
    


