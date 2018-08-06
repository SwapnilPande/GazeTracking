import utils.DataProcessor
import json
import math

#Define ML parameters
with open('ml_param.json') as f: #Open paramter file
	dataPathJSON = json.load(f)['dataPaths'] #Load JSON as dict

pathToData = dataPathJSON['pathToData']
pathTemp = dataPathJSON['pathToTempDir']

batchSize = 10

utils.DataProcessor.initializeData(pathToData,pathTemp, 0.5, 0.25)

trainGenerator = utils.DataProcessor.DataPreProcessor(pathToData, pathTemp, batchSize, 'train', debug = True)
validateGenerator = utils.DataProcessor.DataPreProcessor(pathToData, pathTemp, batchSize, 'validate', debug = True)
testGenerator = utils.DataProcessor.DataPreProcessor(pathToData, pathTemp, batchSize, 'test', debug = True)

try:
	crapGen = utils.DataProcessor.DataPreProcessor(pathToData, pathTemp, batchSize, 'other')
except ValueError:
	print("Successfully caught error for creating invalid dataset")

numBatches = trainGenerator.__len__()
assert numBatches == 20
frames = []
batches = []
for index in range(0, numBatches):
	dataDict, labels, meta = trainGenerator.__getitem__(index)

	if(index != numBatches-1):
		assert len(dataDict['input_1']) == batchSize
		assert len(dataDict['input_2']) == batchSize
		assert len(dataDict['input_3']) == batchSize
		assert len(dataDict['input_4']) == batchSize
		assert len(labels) == batchSize
	else:
		assert len(dataDict['input_1']) == 1
		assert len(dataDict['input_2']) == 1
		assert len(dataDict['input_3']) == 1
		assert len(dataDict['input_4']) == 1
		assert len(labels) == 1
	frames.extend(meta)

if(len(frames) == len(set(frames))):
	print("All frames in epoch are unique")

trainGenerator.on_epoch_end()
frames2 = []
for index in range(0, numBatches):
	__, __, meta = trainGenerator.__getitem__(index)
	frames2.extend(meta)

if(frames != frames2 and set(frames) == set(frames2)):
	print("Second epoch passed")


frames3 = []
for index in range(0, numBatches):
	__, __, meta = validateGenerator.__getitem__(index)
	frames3.extend(meta)

if(frames2 == frames3 or (set(frames2) & set(frames3)) != set()):
	print(set(frames2) & set(frames3))
	print("Frames 3 failed")
