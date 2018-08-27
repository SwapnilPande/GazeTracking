#Import necessary layers for model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Concatenate, Reshape, ZeroPadding2D
#Import initializers for weights and biases
from keras.initializers import Zeros, RandomNormal
from keras.models import Model
from keras.layers.normalization import BatchNormalization

########################## Function definitions for defining model ##########################
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


def createFullyConnected(units, activation = 'relu'):
        return Dense(
                units,
                activation = activation,
                use_bias = True,
                kernel_initializer = randNormKernelInitializer(),
                bias_initializer = 'zeros'
                )
def createBN():
        return  BatchNormalization()

def initializeModel():
        print("Initializing Model")
        #Defining input here
        leftEyeInput = Input(shape=(95,95,3,))
        rightEyeInput = Input(shape=(95,95,3,))
        faceInput = Input(shape=(227,227,3,))
        faceGridInput = Input(shape=(625,))

        #Define convolutional layers for left and right eye inputs
        convE1 = createConvLayer(96, 3, 2)
        maxPoolE1 = createMaxPool()
        BNE1 =createBN()
        paddingE1 = createPadding(1)
        convE2 = createConvLayer(256, 3, 1)
        maxPoolE2 = createMaxPool()
        paddingE2 = createPadding(1)
        convE3 = createConvLayer(384, 3, 1)
        paddingE3 = createPadding(1)
        convE4 = createConvLayer(64, 3, 1)
        maxPoolE3 = createMaxPool()

        #Define convolutional layers for face input
        convF1 = createConvLayer(96, 11, 4)
        maxPoolF1 = createMaxPool()
        BNF1 =createBN()
        paddingF1 = createPadding(2)
        convF2 = createConvLayer(256, 5, 1)
        maxPoolF2 = createMaxPool()
        BNF2 =createBN()
        paddingF2 = createPadding(1)
        convF3 = createConvLayer(384, 3, 1)
        convF4 = createConvLayer(64, 1, 1)
        maxPoolF3 = createMaxPool()

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
        fullyConnected2 = createFullyConnected(2, 'linear')


        #Defining dataflow through layers
        #Left Eye
        leftDataConvE1 = convE1(leftEyeInput)
        leftDataMaxPoolE1 = maxPoolE1(leftDataConvE1)
        leftDataBNE1= BNE1(leftDataMaxPoolE1)
        leftDataPaddingE1 = paddingE1(leftDataBNE1)
        leftDataConvE2 = convE2(leftDataPaddingE1)
        leftDataMaxPoolE2 = maxPoolE2(leftDataConvE2)
        leftDataPaddingE2 = paddingE2(leftDataMaxPoolE2)
        leftDataConvE3 = convE3(leftDataPaddingE2)
        leftDataPaddingE3 = paddingE3(leftDataConvE3)
        leftDataConvE4 = convE4(leftDataPaddingE3)
        leftDataMaxPoolE3=maxPoolE3(leftDataConvE4)
        #Reshape data to feed into fully connected layer
        leftEyeFinal = Reshape((1600,))(leftDataMaxPoolE3)

        #Right Eye
        rightDataConvE1 = convE1(rightEyeInput)
        rightDataMaxPoolE1 = maxPoolE1(rightDataConvE1)
        rightDataBNE1= BNE1(rightDataMaxPoolE1)
        rightDataPaddingE1 = paddingE1(rightDataBNE1)
        rightDataConvE2 = convE2(rightDataPaddingE1)
        rightDataMaxPoolE2 = maxPoolE2(rightDataConvE2)
        rightDataPaddingE2 = paddingE2(rightDataMaxPoolE2)
        rightDataConvE3 = convE3(rightDataPaddingE2)
        rightDataPaddingE3 = paddingE3(rightDataConvE3)
        rightDataConvE4 = convE4(rightDataPaddingE3)
        rightDataMaxPoolE3=maxPoolE3(rightDataConvE4)
        #Reshape data to feed into fully connected layer
        rightEyeFinal = Reshape((1600,))(rightDataMaxPoolE3)

        #Combining left & right eye
        dataLRMerge = Concatenate(axis=1)([leftEyeFinal, rightEyeFinal])
        dataFullyConnectedE1 = fullyConnectedE1(dataLRMerge)

        #Face
        dataConvF1 = convF1(faceInput)
        dataMaxPoolF1 = maxPoolF1(dataConvF1)
        dataBNF1= BNF1(dataMaxPoolF1)
        dataPaddingF1 = paddingF1(dataBNF1)
        dataConvF2 = convF2(dataPaddingF1)
        dataMaxPoolF2 = maxPoolF2(dataConvF2)
        dataBNF2= BNF2(dataMaxPoolF2)
        dataPaddingF2 = paddingF2(dataBNF2)
        dataConvF3 = convF3(dataPaddingF2)
        dataConvF4 = convF4(dataConvF3)
        dataMaxPoolF3 = maxPoolF3(dataConvF4)
        #Reshape data to feed into fully connected layer
        faceFinal = Reshape((2304,))(dataMaxPoolF3)
        dataFullyConnectedF1 = fullyConnectedF1(faceFinal)
        dataFullyConnectedF2 = fullyConnectedF2(dataFullyConnectedF1)


        #Face grid
        dataFullyConnectedFG1 = fullyConnectedFG1(faceGridInput)
        dataFullyConnectedFG2 = fullyConnectedFG2(dataFullyConnectedFG1)

        #Combining Eyes & Face & Face Grid
        finalMerge = Concatenate(axis=1)([dataFullyConnectedE1, dataFullyConnectedF2, dataFullyConnectedFG2])
        dataFullyConnected1 = fullyConnected1(finalMerge)
        finalOutput = fullyConnected2(dataFullyConnected1)


        #Return the fully constructed model
        return Model(inputs = [leftEyeInput, rightEyeInput, faceInput, faceGridInput], outputs = finalOutput)
