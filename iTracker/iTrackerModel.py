#Import necessary layers for model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Concatenate, Reshape, ZeroPadding2D,Activation,SeparableConv2D,AveragePooling2D,Flatten,DepthwiseConv2D,ReLU,Dropout
#Import initializers for weights and biases
from keras.initializers import Zeros, RandomNormal
from keras.models import Model
from keras.layers.normalization import BatchNormalization
import numpy as np

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

def ReLU6(input):
        return ReLU(max_value=6)(input)

def createBottleneck(input,input_channel,output_channel,stride,expansion,padding='same',kernelSize=3):
        output1 = Conv2D(filters=input_channel*expansion,
                         kernel_size=1,
                         strides=(1, 1),
                         padding=padding,
                         data_format=None,
                         dilation_rate=(1, 1),
                         activation=None,
                         use_bias=True,
                         kernel_initializer=randNormKernelInitializer(),
                         bias_initializer='zeros'
                         )(input);
        output1 = createBN(output1);
        output2= ReLU6(output1);
        
        output3= DepthwiseConv2D(kernel_size=kernelSize,
                                  strides=stride,
                                  padding='same',
                                  depth_multiplier=1,
                                  activation=None,
                                  use_bias=True,
                                  )(output2);
        output3 = createBN(output3);
        output4 = ReLU6(output3);

        output5 = Conv2D(filters=output_channel,
                         kernel_size=1,
                         strides=(1, 1),
                         padding=padding,
                         data_format=None,
                         dilation_rate=(1, 1),
                         activation='linear',
                         use_bias=True,
                         kernel_initializer=randNormKernelInitializer(),
                         bias_initializer='zeros'
                         )(output4);
        return output5

def createEyeModelV2(input):
        ## standard ConV+BN+activation
        F1 = createCv(input,128,5,4,padding='same')  #3@224x224 -->> 128@56x56
        F2 = createBN(F1)
        F3 = createActivation(F2)
        F4 = createMaxPool(F3)                       # 128@56x56  -->> 128@27x27
        ## bottleneck block
        F5  = createBottleneck(input=F4,input_channel=128,output_channel=128,kernelSize=3,stride=1,expansion=1,padding='same')          #128@27x27  -->> 128@27x27
        F6 = createMaxPool(F5)                       #128@27x27  -->> 128@13x13
        F7  = createBottleneck(input=F6,input_channel=128,output_channel=256,kernelSize=3,stride=2,expansion=1,padding='same')          #128@13x13  -->> 128@7x7
        F8  = createBottleneck(input=F7,input_channel=256,output_channel=512,kernelSize=3,stride=1,expansion=1,padding='same')          #256@7x7  -->> 512@7x7
        F9  = createMaxPool(F8)                      #512@7x7  -->> 512@3x3
        F10 = Flatten()(F9)
                
        return F10

def createFaceModelV2(input):
        ## standard ConV+BN+activation
        F1 = createCv(input,128,5,4,padding='same')
        F2 = createBN(F1)
        F3 = createActivation(F2)
        F4 = createMaxPool(F3)
        ## bottleneck block
        F5  = createBottleneck(input=F4,input_channel=128,output_channel=128,kernelSize=3,stride=1,expansion=1,padding='same')          #128@27x27  -->> 128@27x27
        F6 = createMaxPool(F5)                       #128@27x27  -->> 128@13x13
        F7  = createBottleneck(input=F6,input_channel=128,output_channel=256,kernelSize=3,stride=2,expansion=1,padding='same')          #128@13x13  -->> 128@7x7
        F8  = createBottleneck(input=F7,input_channel=256,output_channel=512,kernelSize=3,stride=1,expansion=1,padding='same')          #256@7x7  -->> 512@7x7
        F9  = createMaxPool(F8)                      #512@7x7  -->> 512@3x3
        F10 = Flatten()(F9)

        F11 = createFullyConnected(F10,128)
        F11 = Dropout(0.5)(F11)
        F12 = createFullyConnected(F11,64)
        F12 = Dropout(0.5)(F12)
        
        return F12

def createSmallEyeModel(input):
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
def createEyeModel(input):
        ## standard ConV+BN+activation
        F1 = createCv(input,128,5,4,padding='same')  #3@224x224 -->> 128@56x56
        F2 = createBN(F1)
        F3 = createActivation(F2)
        F4 = createMaxPool(F3)                       # 128@56x56  -->> 128@27x27
        ## depthwise separable Conv
        F5  = createDS(F4,128,5,1,1,'same')          #128@27x27  -->> 128@27x27
        F6 = createMaxPool(F5)                       #128@27x27  -->> 128@13x13
        F7  = createDS(F6,256,3,2,1,'same')          #128@13x13  -->>  256@7x7
        F8  = createDS(F7,512,3,1,1,'same')          #256@7x7  -->> 512@7x7 
        F9  = createMaxPool(F8)                      #512@7x7  -->> 512@3x3
        F10 = Flatten()(F9)
        return F10
        
def createFaceModel(input):
        ## standard ConV+BN+activation
        F1 = createCv(input,128,5,4,padding='same')
        F2 = createBN(F1)
        F3 = createActivation(F2)
        F4 = createMaxPool(F3)
        ## depthwise separable Conv
        F5  = createDS(F4,128,5,1,1,'same')
        F6 = createMaxPool(F5)
        F7  = createDS(F6,256,3,2,1,'same')
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

def createEyeLocationModel(input):
        EL1 = createFullyConnected(input,512)
        EL2 = createFullyConnected(EL1, 256)
        return EL2
def createMarkerModel(input):
        MM0 = createFullyConnected(input,128)
        MM1 = createFullyConnected(MM0,256)
        MM1 = Dropout(0.5)(MM1)
        MM2 = createFullyConnected(MM1,512)
        MM2 = Dropout(0.5)(MM2)
        MM3 = createFullyConnected(MM2,256)
        MM3 = Dropout(0.5)(MM3)
        MM4 = createFullyConnected(MM3,128)
        MM4 = Dropout(0.5)(MM4)
        return MM4

def initializeModel():
        
        print("Initializing Model")
        #Defining input here
        leftEyeInput = Input(shape=(224,224,3,))
        rightEyeInput = Input(shape=(224,224,3,))
        faceInput = Input(shape=(224,224,3,))
#        faceGridInput = Input(shape=(6400,))
#        EyeLocationInput = Input(shape=(8,))
        MarkerInput = Input(shape=(10,))
        
        ### eye models
        leftEyeData = createEyeModelV2(leftEyeInput)
        rightEyeData= createEyeModelV2(rightEyeInput)
        faceData = createFaceModelV2(faceInput)
#        faceGridData=createFaceGridModel(faceGridInput)
        markerData = createMarkerModel(MarkerInput)
        
        EyeMerge =  Concatenate(axis=1)([leftEyeData,rightEyeData])
        EyeFc1  = createFullyConnected(EyeMerge,128)
        EyeFc1  = Dropout(0.5)(EyeFc1)

        
        #Combining left & right eye face and faceGrid
        #dataLRMerge = Concatenate(axis=1)([EyeFc1,markerData])
        dataLRMerge = Concatenate(axis=1)([EyeFc1,faceData,markerData])
        dataFc1 = createFullyConnected(dataLRMerge,128)
        dataFc1 = Dropout(0.5)(dataFc1)
        finalOutput = createFullyConnected(dataFc1,2,activation = 'linear')


        #Return the fully constructed model
        #return Model(inputs = [leftEyeInput, rightEyeInput,  MarkerInput], outputs = finalOutput)
        return Model(inputs = [leftEyeInput, rightEyeInput, faceInput, MarkerInput], outputs = finalOutput)
