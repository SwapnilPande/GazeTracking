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
def createCv( filters, kernelSize, stride,padding='same'):
        return Conv2D(
                filters,
                kernelSize, 
                strides = stride,
                activation = None,
                use_bias = True,
                kernel_initializer = randNormKernelInitializer(),
                bias_initializer = 'zeros',
                padding=padding
                )
def createDw(filters, kernelSize, stride,depth_multiplier=1,padding ='same'):
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
                )
# createMaxPool
# Function to simplify the process of creating a MaxPooling layer
# Populates parameters that are common for all maxpool layers in net
# Returns a MaxPooling2D object describing the new layer
def createMaxPool():
        return MaxPooling2D(pool_size = 3, strides = 2)

def createAvePool(pool_size,stride):
        return AveragePooling2D(pool_size = pool_size, strides = stride)

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
def createActivation(activation ='relu'):
        return Activation(activation)

#def createDS(input,filter1,kernelSize1,stride1,padding1,filter2,kernelSize2,stride2,padding2):
#        output1 = createDw(input,filter1,kernelSize1,stride1,padding1)
#        output2 = createBN(output1)
#        output3 = createActivation(output2)
#        output4 = createCv(output3,filter2,kernelSize2,stride2,padding2)
#        output5 = createBN(output4)
#        output6 = createActivation(output5)
#        return output6


def initializeModel():
        print("Initializing Model")
        #Defining input here
        leftEyeInput = Input(shape=(224,224,3,))
        rightEyeInput = Input(shape=(224,224,3,))
        faceInput = Input(shape=(224,224,3,))
        faceGridInput = Input(shape=(625,))

        ### eye models
        EBN1  = createBN()
        EBN2  = createBN()
        EBN3  = createBN()
        EBN4  = createBN()
        EBN5  = createBN()
        EBN6  = createBN()
        EBN7  = createBN()
        EBN8  = createBN()
        EBN9  = createBN()
        EBN10 = createBN()
        EBN11 = createBN()
        EBN12 = createBN()
        EBN13 = createBN()
        EBN14 = createBN()
        EBN15 = createBN()
        EBN16 = createBN()
        EBN17 = createBN()
        EBN18 = createBN()
        EBN19 = createBN()
        EBN20 = createBN()
        EBN21 = createBN()
        EBN22 = createBN()
        EBN23 = createBN()
        EBN24 = createBN()
        EBN25 = createBN()
        EBN26 = createBN()
        EBN27 = createBN()

        EDW1  = createDw(32,3,1,2)
        EDW2  = createDw(64,3,2,2)
        EDW3  = createDw(128,3,1,1)
        EDW4  = createDw(128,3,2,2)
        EDW5  = createDw(256,3,1,1)
        EDW6  = createDw(256,3,2,2)
        EDW7  = createDw(512,3,1,1)
        EDW8  = createDw(512,3,1,1)
        EDW9  = createDw(512,3,1,1)
        EDW10 = createDw(512,3,1,1)
        EDW11 = createDw(512,3,1,1)
        EDW12 = createDw(512,3,2,2)
        EDW13 = createDw(1024,3,1,1)

        ECV1  = createCv(32,3,2,padding='same')


        ReluActivation = createActivation()
        EAvePool = createAvePool(7,1)
        MaxPool = createMaxPool()
        EFC1 = createFullyConnected(256)
        EFC2 = createFullyConnected(128)

        ### face models
        FBN1  = createBN()
        FBN2  = createBN()
        FBN3  = createBN()
        FBN4  = createBN()
        FBN5  = createBN()
        FBN6  = createBN()
        FBN7  = createBN()
        FBN8  = createBN()
        FBN9  = createBN()
        FBN10 = createBN()
        FBN11 = createBN()
        FBN12 = createBN()
        FBN13 = createBN()
        FBN14 = createBN()
        FBN15 = createBN()
        FBN16 = createBN()
        FBN17 = createBN()
        FBN18 = createBN()
        FBN19 = createBN()
        FBN20 = createBN()
        FBN21 = createBN()
        FBN22 = createBN()
        FBN23 = createBN()
        FBN24 = createBN()
        FBN25 = createBN()
        FBN26 = createBN()
        FBN27 = createBN()

        
        FDW1  = createDw(32,3,1,2)
        FDW2  = createDw(64,3,2,2)
        FDW3  = createDw(128,3,1,1)
        FDW4  = createDw(128,3,2,2)
        FDW5  = createDw(256,3,1,1)
        FDW6  = createDw(256,3,2,2)
        FDW7  = createDw(512,3,1,1)
        FDW8  = createDw(512,3,1,1)
        FDW9  = createDw(512,3,1,1)
        FDW10 = createDw(512,3,1,1)
        FDW11 = createDw(512,3,1,1)
        FDW12 = createDw(512,3,2,2)
        FDW13 = createDw(1024,3,1,1)
        
        FCV1  = createCv(32,3,2,padding='same')


       # ReluActivation = createActivation()
        FAvePool = createAvePool(7,1)
        FFC1 = createFullyConnected(256)
        FFC2 = createFullyConnected(128)

        ###face grid layers

        FGFC1 = createFullyConnected( 256)
        FGFC2 = createFullyConnected( 256)

        ### data flow
        ## left eye
        LE1 = ECV1(leftEyeInput)
        LE2 = EBN1(LE1)
        LE3 = ReluActivation(LE2)

        LE4 = EDW1(LE3)
        LE5 = EBN2(LE4)
        LE6 = ReluActivation(LE5)

        LE7 = EDW2(LE6)
        LE8 = EBN4(LE7)
        LE9 = ReluActivation(LE8)
        
        LE10 = EDW3(LE9)
        LE11 = EBN6(LE10)
        LE12 = ReluActivation(LE11)

        LE13 = EDW4(LE12)
        LE14 = EBN8(LE13)
        LE15 = ReluActivation(LE14)

        LE16 = EDW5(LE15)
        LE17 = EBN10(LE16)
        LE18 = ReluActivation(LE17)

        LE19 = EDW6(LE18)
        LE20 = EBN12(LE19)
        LE21 = ReluActivation(LE20)
        
        LE22 = EDW7(LE21)
        LE23 = EBN14(LE22)
        LE24 = ReluActivation(LE23)
        
        LE25 = EDW8(LE24)
        LE26 = EBN16(LE25)
        LE27 = ReluActivation(LE26)
        
        LE28 = EDW9(LE27)
        LE29 = EBN18(LE28)
        LE30 = ReluActivation(LE29)
        
        LE31 = EDW10(LE30)
        LE32 = EBN20(LE31)
        LE33 = ReluActivation(LE32)
        
        LE34 = EDW11(LE33)
        LE35 = EBN22(LE34)
        LE36 = ReluActivation(LE35)
        
        LE37 = EDW12(LE36)
        LE38 = EBN24(LE37)
        LE39 = ReluActivation(LE38)
        
        LE40 = EDW13(LE39)
        LE41 = EBN26(LE40)
        LE42 = ReluActivation(LE41)

        LE43 = EAvePool(LE42)
        LE44 = Flatten()(LE43)
        LE45 = EFC1(LE44)
        LE46 = EFC2(LE45)
        ## right eye
        
        RE1 = ECV1(rightEyeInput)
        RE2 = EBN1(RE1)
        RE3 = ReluActivation(RE2)

        RE4 = EDW1(RE3)
        RE5 = EBN2(RE4)
        RE6 = ReluActivation(RE5)

        RE7 = EDW2(RE6)
        RE8 = EBN4(RE7)
        RE9 = ReluActivation(RE8)
        
        RE10 = EDW3(RE9)
        RE11 = EBN6(RE10)
        RE12 = ReluActivation(RE11)

        RE13 = EDW4(RE12)
        RE14 = EBN8(RE13)
        RE15 = ReluActivation(RE14)

        RE16 = EDW5(RE15)
        RE17 = EBN10(RE16)
        RE18 = ReluActivation(RE17)

        RE19 = EDW6(RE18)
        RE20 = EBN12(RE19)
        RE21 = ReluActivation(RE20)
        
        RE22 = EDW7(RE21)
        RE23 = EBN14(RE22)
        RE24 = ReluActivation(RE23)
        
        RE25 = EDW8(RE24)
        RE26 = EBN16(RE25)
        RE27 = ReluActivation(RE26)
        
        RE28 = EDW9(RE27)
        RE29 = EBN18(RE28)
        RE30 = ReluActivation(RE29)
        
        RE31 = EDW10(RE30)
        RE32 = EBN20(RE31)
        RE33 = ReluActivation(RE32)
        
        RE34 = EDW11(RE33)
        RE35 = EBN22(RE34)
        RE36 = ReluActivation(RE35)
        
        RE37 = EDW12(RE36)
        RE38 = EBN24(RE37)
        RE39 = ReluActivation(RE38)
        
        RE40 = EDW13(RE39)
        RE41 = EBN26(RE40)
        RE42 = ReluActivation(RE41)

        RE43 = EAvePool(RE42)
        RE44 = Flatten()(RE43)
        RE45 = EFC1(RE44)
        RE46 = EFC2(RE45)


        ## face
        
        FC1 = FCV1(faceInput)
        FC2 = FBN1(FC1)
        FC3 = ReluActivation(FC2)

        FC4 = FDW1(FC3)
        FC5 = FBN2(FC4)
        FC6 = ReluActivation(FC5)

        FC7 = FDW2(FC6)
        FC8 = FBN4(FC7)
        FC9 = ReluActivation(FC8)
        
        FC10 = FDW3(FC9)
        FC11 = FBN6(FC10)
        FC12 = ReluActivation(FC11)

        FC13 = FDW4(FC12)
        FC14 = FBN8(FC13)
        FC15 = ReluActivation(FC14)

        FC16 = FDW5(FC15)
        FC17 = FBN10(FC16)
        FC18 = ReluActivation(FC17)

        FC19 = FDW6(FC18)
        FC20 = FBN12(FC19)
        FC21 = ReluActivation(FC20)
        
        FC22 = FDW7(FC21)
        FC23 = FBN14(FC22)
        FC24 = ReluActivation(FC23)
        
        FC25 = FDW8(FC24)
        FC26 = FBN16(FC25)
        FC27 = ReluActivation(FC26)
        
        FC28 = FDW9(FC27)
        FC29 = FBN18(FC28)
        FC30 = ReluActivation(FC29)
        
        FC31 = FDW10(FC30)
        FC32 = FBN20(FC31)
        FC33 = ReluActivation(FC32)
        
        FC34 = FDW11(FC33)
        FC35 = FBN22(FC34)
        FC36 = ReluActivation(FC35)
        
        FC37 = FDW12(FC36)
        FC38 = FBN24(FC37)
        FC39 = ReluActivation(FC38)
        
        FC40 = FDW13(FC39)
        FC41 = FBN26(FC40)
        FC42 = ReluActivation(FC41)

        FC43 = EAvePool(FC42)
        FC44 = Flatten()(FC43)
        FC45 = FFC1(FC44)
        FC46 = FFC2(FC45)

        #Face Grid
        FG1 = FGFC1(faceGridInput)
        FG2 = FGFC2(FG1)
        
        #Combining left & right eye face and faceGrid
        dataLRMerge = Concatenate(axis=1)([LE46,RE46,FC46,FG2])
        dataFc1 = createFullyConnected(128)(dataLRMerge)
        finalOutput = createFullyConnected(2,activation = 'linear')(dataFc1)


        #Return the fully constructed model
        return Model(inputs = [leftEyeInput, rightEyeInput, faceInput, faceGridInput], outputs = finalOutput)
