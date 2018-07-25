from uiUtils import yesNoPrompt
#Custom datset processor
from processData import DataPreProcessor

with open('ml_param.json') as f: #Open paramter file
	paramJSON = json.load(f) #Load JSON as dict

dataPathJSON = paramJSON['dataPaths'] # Extract datapath information
pathToData = dataPathJSON['pathToData']
pathTemp = dataPathJSON['pathToTempDir']

#Initialize Data pre-processor here
pp = DataPreProcessor(pathToData, pathTemp, trainSetProportion, validateSetProportion)