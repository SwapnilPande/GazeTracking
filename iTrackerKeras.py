#TODO Determine how to use learning rate multipliers
#TODO Determine how to use grouping in convolutional layer
#TODO Determine how to create LRN layer
from keras.models import Model

#Import necessary layers for model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Concatenate, Reshape, ZeroPadding2D

#Import initializers for weights and biases
from keras.initializers import Zeros, RandomNormal

from keras.optimizers import SGD

#Defining input here
leftEyeInput = Input(shape=(224,224,3,))
rightEyeInput = Input(shape=(224,224,3,))
faceInput = Input(shape=(224,224,3,))
faceGridInput = Input(shape=(256,))

def randNormKernelInitializer():
	return RandomNormal(stddev= 0.01)

# createConvLayer
# Function to simplify the process of creating a convolutional layer for iCapture
# Populates parameters that are common for all convolutional layers in network
#
# INPUTS
# filters - Number of feature layers in the output
# kernel_size - dimension of kernel in pixels - creates square kernel
# stride - Stride taken during convolution
#
# Returns a Conv2D object describing the new layer
def createConvLayer(filters, kernelSize, stride):
	return Conv2D(
		filters,
		kernelSize, 
		strides = stride,
		activation = 'relu',
		use_bias = True,
		kernel_initializer = randNormKernelInitializer(),
		bias_initializer = 'zeros'
		)

# createMaxPool
# Function to simplify the process of creating a MaxPooling layer
# Populates parameters that are common for all maxpool layers in net
# Returns a MaxPooling2D object describing the new layer
def createMaxPool():
	return MaxPooling2D(pool_size = 3, strides = 2)

def createPadding(pad):
	return ZeroPadding2D(padding=pad)


def createFullyConnected(units):
	return Dense(
		units,
	 	activation = 'relu',
	 	use_bias = True,
	 	kernel_initializer = randNormKernelInitializer(),
		bias_initializer = 'zeros'
	 	)


#Define convolutional layers for left and right eye inputs
convE1 = createConvLayer(96, 11, 4)
maxPoolE1 = createMaxPool()
paddingE1 = createPadding(2)
convE2 = createConvLayer(256, 5, 1)
maxPoolE2 = createMaxPool()
paddingE2 = createPadding(1)
convE3 = createConvLayer(384, 3, 1)
convE4 = createConvLayer(64, 1, 1)

#Define convolutional layers for face input
convF1 = createConvLayer(96, 11, 4)
maxPoolF1 = createMaxPool()
paddingF1 = createPadding(2)
convF2 = createConvLayer(256, 5, 1)
maxPoolF2 = createMaxPool()
paddingF2 = createPadding(1)
convF3 = createConvLayer(384, 3, 1)
convF4 = createConvLayer(64, 1, 1)

#Define fully connected layer for left & right eye concatenation
fullyConnectedE1 = createFullyConnected(128)

#Define fully connected layers for face
fullyConnectedF1 = createFullyConnected(128)
fullyConnectedF2 = createFullyConnected(64)

#Define fully connected layers for face grid
fullyConnectedFG1 = createFullyConnected(256)
fullyConnectedFG2 = createFullyConnected(128)

#Define fully connected layers for eyes & face & face grid
fullyConnected1 = createFullyConnected(128)
fullyConnected2 = createFullyConnected(2)


#Defining dataflow through layers
#Left Eye
leftDataConvE1 = convE1(leftEyeInput)
leftDataMaxPoolE1 = maxPoolE1(leftDataConvE1)
leftDataPaddingE1 = paddingE1(leftDataMaxPoolE1)
leftDataConvE2 = convE2(leftDataPaddingE1)
leftDataMaxPoolE2 = maxPoolE2(leftDataConvE2)
leftDataPaddingE2 = paddingE2(leftDataMaxPoolE2)
leftDataConvE3 = convE3(leftDataPaddingE2)
leftDataConvE4 = convE4(leftDataConvE3)
#Reshape data to feed into fully connected layer
leftEyeFinal = Reshape((9216,))(leftDataConvE4)

#Right Eye
rightDataConvE1 = convE1(rightEyeInput)
rightDataMaxPoolE1 = maxPoolE1(rightDataConvE1)
rightDataPaddingE1 = paddingE1(rightDataMaxPoolE1)
rightDataConvE2 = convE2(rightDataPaddingE1)
rightDataMaxPoolE2 = maxPoolE2(rightDataConvE2)
rightDataPaddingE2 = paddingE2(rightDataMaxPoolE2)
rightDataConvE3 = convE3(rightDataPaddingE2)
rightDataConvE4 = convE4(rightDataConvE3)
#Reshape data to feed into fully connected layer
rightEyeFinal = Reshape((9216,))(rightDataConvE4)

#Combining left & right eye
dataLRMerge = Concatenate(axis=1)([leftEyeFinal, rightEyeFinal])
dataFullyConnectedE1 = fullyConnectedE1(dataLRMerge)

#Face
dataConvF1 = convF1(faceInput)
dataMaxPoolF1 = maxPoolF1(dataConvF1)
dataPaddingF1 = paddingF1(dataMaxPoolF1)
dataConvF2 = convF2(dataPaddingF1)
dataMaxPoolF2 = maxPoolF2(dataConvF2)
dataPaddingF2 = paddingF2(dataMaxPoolF2)
dataConvF3 = convF3(dataPaddingF2)
dataConvF4 = convF4(dataConvF3)
#Reshape data to feed into fully connected layer
faceFinal = Reshape((9216,))(dataConvF4)
dataFullyConnectedF1 = fullyConnectedF1(faceFinal)
dataFullyConnectedF2 = fullyConnectedF2(dataFullyConnectedF1)


#Face grid
dataFullyConnectedFG1 = fullyConnectedFG1(faceGridInput)
dataFullyConnectedFG2 = fullyConnectedFG2(dataFullyConnectedFG1)

#Combining Eyes & Face & Face Grid
finalMerge = Concatenate(axis=1)([dataFullyConnectedE1, dataFullyConnectedF2, dataFullyConnectedFG2])
dataFullyConnected1 = fullyConnected1(finalMerge)
finalOutput = fullyConnected2(dataFullyConnected1)

#Initializing the model
iTrackerModel = Model(inputs = [leftEyeInput, rightEyeInput, faceInput, faceGridInput], outputs = finalOutput)

def getSGDOptimizer():
	return SGD(lr=0.001, momentum=0.9, decay=0.0005)

iTrackerModel.compile(getSGDOptimizer(), loss=['mean_squared_error'], metrics=['accuracy'])





