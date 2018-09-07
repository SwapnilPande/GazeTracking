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
def createMaxPool(input,size =3):
        return MaxPooling2D(pool_size = size, strides = 2)(input)

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
        E1 = createCv(input,64,5,1,padding='same')    #3@60x60 -->> 64@60x60
        E2 = createBN(E1)
        E3 = createActivation(E2)
        E4 = createMaxPool(E3,2)                      #64@60x60 -->> 64@30x30
        ## depthwise separable Conv
        E5  = createDS(E4,128,5,1,1,'same')           #64@30x30 -->>  128@30x30
        E6 = createMaxPool(E5,2)                      #128@30x30 -->> 128@15x15
        E7  = createDS(E6,256,3,1,1,'same')          # 128@15x15 -->> 256@15x15
#        E8  = createDS(E7,512,3,1,1,'same')
        E9  = createMaxPool(E7,3)                    #256@15x15 -->>256@7x7
        E10 = Flatten()(E9)
        
        return E10
        
def createFaceModel(input):
        ## standard ConV+BN+activation
        F1 = createCv(input,128,5,4,padding='same')
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
def initializeModel():
        
        print("Initializing Model")
        #Defining input here
        leftEyeInput = Input(shape=(60,60,3,))
        rightEyeInput = Input(shape=(60,60,3,))
        faceInput = Input(shape=(224,224,3,))
        faceGridInput = Input(shape=(625,))

        ### eye models
        leftEyeData = createEyeModel(leftEyeInput)
        rightEyeData= createEyeModel(rightEyeInput)
        faceData = createFaceModel(faceInput)
        faceGridData=createFaceGridModel(faceGridInput)

        EyeMerge =  Concatenate(axis=1)([leftEyeData,rightEyeData])
        EyeFc1  = createFullyConnected(EyeMerge,128)

        
        #Combining left & right eye face and faceGrid
        dataLRMerge = Concatenate(axis=1)([EyeFc1,faceData,faceGridData])
        dataFc1 = createFullyConnected(dataLRMerge,128)
        finalOutput = createFullyConnected(dataFc1,2,activation = 'linear')


        #Return the fully constructed model
        return Model(inputs = [leftEyeInput, rightEyeInput, faceInput, faceGridInput], outputs = finalOutput)
