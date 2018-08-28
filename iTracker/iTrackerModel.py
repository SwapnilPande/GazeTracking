#Import necessary layers for model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Concatenate, Reshape, ZeroPadding2D,Activation,SeparableConv2D,AveragePooling2D,Flatten
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
def createCv(input, filters, kernelSize, stride,padding='valid'):
        return Conv2D(
                filters,
                kernelSize, 
                strides = stride,
                activation = None,
                use_bias = True,
                kernel_initializer = randNormKernelInitializer(),
                bias_initializer = 'zeros',
                padding=padding
                )(input)
def createDw(input,filters, kernelSize, stride,padding ='valid'):
        return SeparableConv2D(
                filters,
                kernelSize, 
                strides = stride,
                activation = None,
                use_bias = True,
                kernel_initializer = randNormKernelInitializer(),
                bias_initializer = 'zeros',
                padding =padding
                )(input)
# createMaxPool
# Function to simplify the process of creating a MaxPooling layer
# Populates parameters that are common for all maxpool layers in net
# Returns a MaxPooling2D object describing the new layer
def createMaxPool(input):
        return MaxPooling2D(pool_size = 3, strides = 2)(input)

def createAvePool(input,pool_size,stride):
        return AveragePooling2D(pool_size = pool_size, strides = stride)(input)

def createPadding(input,pad):
        return ZeroPadding2D(padding=pad)(input)


def createFullyConnected(input,units, activation = 'relu'):
        return Dense(
                units,
                activation = activation,
                use_bias = True,
                kernel_initializer = randNormKernelInitializer(),
                bias_initializer = 'zeros'
                )(input)
def createBN(input):
        return  BatchNormalization()(input)
def createActivation(input,activation ='relu'):
        return Activation(activation)(input)

def createDS(input,filter1,kernelSize1,stride1,padding1,filter2,kernelSize2,stride2,padding2):
        output1 = createDw(input,filter1,kernelSize1,stride1,padding1)
        output2 = createBN(output1)
        output3 = createActivation(output2)
        output4 = createCv(output3,filter2,kernelSize2,stride2,padding2)
        output5 = createBN(output4)
        output6 = createActivation(output5)
        return output6

def createEyeModel(input):
        ## standard ConV+BN+activation
        E1 = createCv(input,32,3,2,padding='same')
        E2 = createBN(E1)
        E3 = createActivation(E2)
        ## depthwise separable Conv
        E4  = createDS(E3,32,3,1,'same',64,1,1,'valid')
        E5  = createDS(E4,64,3,2,'same',128,1,1,'valid')
        E6  = createDS(E5,128,3,1,'same',128,1,1,'valid')
        E7  = createDS(E6,128,3,2,'same',256,1,1,'valid')
        E8  = createDS(E7,256,3,1,'same',256,1,1,'valid')
        E9  = createDS(E8,256,3,2,'same',512,1,1,'valid')
        E10 = createDS(E9,512,3,1,'same',512,1,1,'valid')
        E11 = createDS(E10,512,3,1,'same',512,1,1,'valid')
        E12 = createDS(E11,512,3,1,'same',512,1,1,'valid')
        E13 = createDS(E12,512,3,1,'same',512,1,1,'valid')
        E14 = createDS(E13,512,3,1,'same',512,1,1,'valid')
        E15 = createDS(E10,512,3,2,'same',1024,1,1,'valid')
        E16 = createDS(E15,1024,3,1,'same',1024,1,1,'valid')

        ##average pool and FC
        E17 = createAvePool(E16,7,1)
        E18 = Flatten()(E17)
        E19 = createFullyConnected(E18,256)

        return E19
        
def createFaceGridModel(input):
        FG1 = createFullyConnected(input, 256)
        FG2 = createFullyConnected(FG1, 256)
        return FG2

def initializeModel():
        print("Initializing Model")
        #Defining input here
        leftEyeInput = Input(shape=(224,224,3,))
        rightEyeInput = Input(shape=(224,224,3,))
        faceInput = Input(shape=(224,224,3,))
        faceGridInput = Input(shape=(625,))

        leftEyeData  = createEyeModel(leftEyeInput)
        rightEyeData = createEyeModel(rightEyeInput)
        faceData     = createEyeModel(faceInput)
        faceGridData = createFaceGridModel(faceGridInput)
        
        #Combining left & right eye face and faceGrid
        dataLRMerge = Concatenate(axis=1)([leftEyeData, rightEyeData,faceData,faceGridData])
        dataFc1 = createFullyConnected(dataLRMerge,128)
        finalOutput = createFullyConnected(dataFc1,2,activation = 'linear')


        #Return the fully constructed model
        return Model(inputs = [leftEyeInput, rightEyeInput, faceInput, faceGridInput], outputs = finalOutput)
