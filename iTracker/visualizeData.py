from uiUtils import yesNoPrompt
#Custom datset processor
from processData import DataPreProcessor
import uiUtils

with open('ml_param.json') as f: #Open paramter file
	paramJSON = json.load(f) #Load JSON as dict

dataPathJSON = paramJSON['dataPaths'] # Extract datapath information
pathToData = dataPathJSON['pathToData']
pathTemp = dataPathJSON['pathToTempDir']

#Initialize Data pre-processor here
processData.initializeData(pathToData, pathTemp, trainSetProportion, validateSetProportion)

#Prompt user to select dataset
print("Which dataset would you like to visualize?")
dataset = uiUtils.yesNoPrompt(['train','validate','test'])


