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
def createDw(filters, kernelSize, stride,padding ='same'):
        return SeparableConv2D(
                filters,
                kernelSize, 
                strides = stride,
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
        leftEyeInput = Input(shape=(112,112,3,))
        rightEyeInput = Input(shape=(112,112,3,))
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

        EDW1  = createDw(64,3,2)
        EDW2  = createDw(128,3,2)
        EDW3  = createDw(256,3,1)
        EDW4  = createDw(512,3,2)

        ECV1  = createCv(64,3,2,padding='same')
        ECV2  = createCv(128,1,1)
        ECV3  = createCv(256,1,1)
        ECV4  = createCv(512,1,1)
        ECV5  = createCv(1024,1,1)

        ReluActivation = createActivation()
        EAvePool = createAvePool(7,1)
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

        FDW1  = createDw(32,3,2)
        FDW2  = createDw(64,3,2)
        FDW3  = createDw(128,3,2)
        FDW4  = createDw(256,3,1)
        FDW5  = createDw(512,3,2)



        
        FCV1  = createCv(32,3,2,padding='same')
        FCV2  = createCv(64,1,1)
        FCV3  = createCv(128,1,1)
        FCV4  = createCv(256,1,1)
        FCV5  = createCv(512,1,1)
        FCV6  = createCv(1024,1,1)


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
        LE7 = ECV2(LE6)
        LE8 = EBN3(LE7)
        LE9 = ReluActivation(LE8)

        LE10 = EDW2(LE9)
        LE11 = EBN4(LE10)
        LE12 = ReluActivation(LE11)
        LE13 = ECV3(LE12)
        LE14 = EBN5(LE13)
        LE15 = ReluActivation(LE14)
        
        LE16 = EDW3(LE15)
        LE17 = EBN6(LE16)
        LE18 = ReluActivation(LE17)
        LE19 = ECV4(LE18)
        LE20 = EBN7(LE19)
        LE21 = ReluActivation(LE20)

        LE22 = EDW4(LE21)
        LE23 = EBN8(LE22)
        LE24 = ReluActivation(LE23)
        LE25 = ECV5(LE24)
        LE26 = EBN9(LE25)
        LE27 = ReluActivation(LE26)

        LE82 = EAvePool(LE27)
        LE83 = Flatten()(LE82)
#        LE84 = EFC1(LE83)
#        LE85 = EFC2(LE84)
        ## right eye
        
        RE1 = ECV1(rightEyeInput)
        RE2 = EBN1(RE1)
        RE3 = ReluActivation(RE2)

        RE4 = EDW1(RE3)
        RE5 = EBN2(RE4)
        RE6 = ReluActivation(RE5)
        RE7 = ECV2(RE6)
        RE8 = EBN3(RE7)
        RE9 = ReluActivation(RE8)

        RE10 = EDW2(RE9)
        RE11 = EBN4(RE10)
        RE12 = ReluActivation(RE11)
        RE13 = ECV3(RE12)
        RE14 = EBN5(RE13)
        RE15 = ReluActivation(RE14)
        
        RE16 = EDW3(RE15)
        RE17 = EBN6(RE16)
        RE18 = ReluActivation(RE17)
        RE19 = ECV4(RE18)
        RE20 = EBN7(RE19)
        RE21 = ReluActivation(RE20)

        RE22 = EDW4(RE21)
        RE23 = EBN8(RE22)
        RE24 = ReluActivation(RE23)
        RE25 = ECV5(RE24)
        RE26 = EBN9(RE25)
        RE27 = ReluActivation(RE26)

        RE82 = EAvePool(RE27)
        RE83 = Flatten()(RE82)
#        RE84 = EFC1(RE83)
#        RE85 = EFC2(RE84)
        EyeMerge = Concatenate(axis =1)([LE83,RE83])
        EyeFC1 = EFC1(EyeMerge)
        EyeFC2 = EFC2(EyeFC1)

        
        ## face
        
        FC1 = FCV1(faceInput)
        FC2 = FBN1(FC1)
        FC3 = ReluActivation(FC2)

        FC4 = FDW1(FC3)
        FC5 = FBN2(FC4)
        FC6 = ReluActivation(FC5)
        FC7 = FCV2(FC6)
        FC8 = FBN3(FC7)
        FC9 = ReluActivation(FC8)

        FC10 = FDW2(FC9)
        FC11 = FBN4(FC10)
        FC12 = ReluActivation(FC11)
        FC13 = FCV3(FC12)
        FC14 = FBN5(FC13)
        FC15 = ReluActivation(FC14)
        
        FC16 = FDW3(FC15)
        FC17 = FBN6(FC16)
        FC18 = ReluActivation(FC17)
        FC19 = FCV4(FC18)
        FC20 = FBN7(FC19)
        FC21 = ReluActivation(FC20)

        FC22 = FDW4(FC21)
        FC23 = FBN8(FC22)
        FC24 = ReluActivation(FC23)
        FC25 = FCV5(FC24)
        FC26 = FBN9(FC25)
        FC27 = ReluActivation(FC26)

        FC28 = FDW5(FC27)
        FC29 = FBN10(FC28)
        FC30 = ReluActivation(FC29)
        FC31 = FCV6(FC30)
        FC32 = FBN11(FC31)
        FC33 = ReluActivation(FC32)

        FC82 = FAvePool(FC33)
        FC83 = Flatten()(FC82)
        FC84 = FFC1(FC83)
        FC85 = FFC2(FC84)

        #Face Grid
        FG1 = FGFC1(faceGridInput)
        FG2 = FGFC2(FG1)
        
        #Combining left & right eye face and faceGrid
        dataLRMerge = Concatenate(axis=1)([EyeFC2,FC84,FG2])
        dataFc1 = createFullyConnected(128)(dataLRMerge)
        finalOutput = createFullyConnected(2,activation = 'linear')(dataFc1)


        #Return the fully constructed model
        return Model(inputs = [leftEyeInput, rightEyeInput, faceInput, faceGridInput], outputs = finalOutput)
