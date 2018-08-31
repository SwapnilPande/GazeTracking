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
def createCv(input, filters, kernelSize, stride,padding='same'):
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
def createDw(input,filters, kernelSize, stride,depth_multiplier=1,padding ='same'):
        return SeparableConv2D(
                filters,
                kernelSize, 
                strides = stride,
                depth_multiplier = depth_multiplier,
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
 
def createAvePool(input, pool_size,stride):
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
 
def createDS(input,filter,kernelSize,stride,depth_multiplier,padding):
        output1 = createDw(input,filter,kernelSize,stride,depth_multiplier,padding)
        output2 = createBN(output1)
        output3 = createActivation(output2)
        return output3


def createEyeModel(input):
        ## standard ConV+BN+activation
        E1 = createCv(input,128,11,4,padding='same')
        E2 = createBN(E1)
        E3 = createActivation(E2)
        E4 = createMaxPool(E3)
        ## depthwise separable Conv
        E5  = createDS(E4,128,5,1,2,'same')
        E6 = createMaxPool(E5)
        E7  = createDS(E6,256,3,2,2,'same')
        E8  = createDS(E7,512,3,1,1,'same')
        E9  = createMaxPool(E8)
        E10 = Flatten()(E9)
        
        return E10
        
def createFaceModel(input):
        ## standard ConV+BN+activation
        F1 = createCv(input,128,11,4,padding='same')
        F2 = createBN(F1)
        F3 = createActivation(F2)
        F4 = createMaxPool(F3)
        ## depthwise separable Conv
        F5  = createDS(F4,128,5,1,2,'same')
        F6 = createMaxPool(F5)
        F7  = createDS(F6,256,3,2,2,'same')
        F8  = createDS(F7,512,3,1,1,'same')
        F9  = createMaxPool(F8)
        F10 = Flatten()(F9)
 
        F11 = createFullyConnected(F10,128)
        F12 = createFullyConnected(F11,64)
        
        return F12
 
def createFaceGridModel(input):
        FG1 = createFullyConnected(input, 256)
        FG2 = createFullyConnected(FG1, 256)
        return FG2

def createFullImageModel(input):
        ## standard ConV+BN+activation
        F1 = createCv(input,64,11,4,padding='same')    # input 320x320x3   output 80x80x64
        F2 = createBN(F1)
        F3 = createActivation(F2)

        ## depthwise separable Conv
        F4  = createDS(F3,64,5,1,1,'same')       # 80x80x64
        F5  = createDS(F4,128,3,2,1,'same')      # 40*40*128
        F6  = createDS(F5,128,3,1,1,'same')      # 40*40*128
        F7  = createDS(F6,256,3,2,1,'same')      # 20x20x256
        F8  = createDS(F7,256,3,1,1,'same')      # 20x20x256
        F9  = createDS(F8,512,3,2,1,'same')      # 10x10x512
        F10  = createDS(F9,512,3,1,1,'same')      # 10x10x512    
        F11  = createDS(F10,512,3,1,1,'same')     # 10x10x512     
        F12  = createDS(F11,512,3,1,1,'same')     # 10x10x512 
        F13  = createDS(F12,512,3,1,1,'same')     # 10x10x512 
        F14  = createDS(F13,512,3,1,1,'same')     # 10x10x512
        F15  = createDS(F14,1024,3,2,1,'same')     # 5x5x1024
        F16  = createDS(F15,1024,3,1,1,'same')     # 5x5x1024 
        F17  = createMaxPool(F16)                   #  3x3x1024
        F18 = Flatten()(F17)
       
        return F18
 
        

def initializeModel():
        print("Initializing Model")
        #Defining input here
        imageInput =Input(shape=(320,320,3,))
        
        fullImageData = createFullImageModel(imageInput)
        dataFullyConnected1 = createFullyConnected(fullImageData,128)
        dataFullyConnected2 = createFullyConnected(dataFullyConnected1,128)
        dataFullyConnected3 = createFullyConnected(dataFullyConnected2,64)
        finalOutput = createFullyConnected(dataFullyConnected3,2,'linear')

        #Return the fully constructed model
        return Model(inputs = imageInput, outputs = finalOutput)
