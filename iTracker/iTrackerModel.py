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

        EDW1  = createDw(32,3,1)
        EDW2  = createDw(64,3,2)
        EDW3  = createDw(128,3,1)
        EDW4  = createDw(128,3,2)
        EDW5  = createDw(256,3,1)
        EDW6  = createDw(256,3,2)
        EDW7  = createDw(512,3,1)
        EDW8  = createDw(512,3,1)
        EDW9  = createDw(512,3,1)
        EDW10 = createDw(512,3,1)
        EDW11 = createDw(512,3,1)
        EDW12 = createDw(512,3,2)
        EDW13 = createDw(1024,3,1)

        ECV1  = createCv(32,3,2,padding='same')
        ECV2  = createCv(64,1,1)
        ECV3  = createCv(128,1,1)
        ECV4  = createCv(128,1,1)
        ECV5  = createCv(256,1,1)
        ECV6  = createCv(256,1,1)
        ECV7  = createCv(512,1,1)
        ECV8  = createCv(512,1,1)
        ECV9  = createCv(512,1,1)
        ECV10 = createCv(512,1,1)
        ECV11 = createCv(512,1,1)
        ECV12 = createCv(512,1,1)
        ECV13 = createCv(1024,1,1)
        ECV14 = createCv(1024,1,1)

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

        FDW1  = createDw(32,3,1)
        FDW2  = createDw(64,3,2)
        FDW3  = createDw(128,3,1)
        FDW4  = createDw(128,3,2)
        FDW5  = createDw(256,3,1)
        FDW6  = createDw(256,3,2)
        FDW7  = createDw(512,3,1)
        FDW8  = createDw(512,3,1)
        FDW9  = createDw(512,3,1)
        FDW10 = createDw(512,3,1)
        FDW11 = createDw(512,3,1)
        FDW12 = createDw(512,3,2)
        FDW13 = createDw(1024,3,1)

        
        FCV1  = createCv(32,3,2,padding='same')
        FCV2  = createCv(64,1,1)
        FCV3  = createCv(128,1,1)
        FCV4  = createCv(128,1,1)
        FCV5  = createCv(256,1,1)
        FCV6  = createCv(256,1,1)
        FCV7  = createCv(512,1,1)
        FCV8  = createCv(512,1,1)
        FCV9  = createCv(512,1,1)
        FCV10 = createCv(512,1,1)
        FCV11 = createCv(512,1,1)
        FCV12 = createCv(512,1,1)
        FCV13 = createCv(1024,1,1)
        FCV14 = createCv(1024,1,1)

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

        LE28 = EDW5(LE27)
        LE29 = EBN10(LE28)
        LE30 = ReluActivation(LE29)
        LE31 = ECV6(LE30)
        LE32 = EBN11(LE31)
        LE33 = ReluActivation(LE32)

        LE34 = EDW6(LE33)
        LE35 = EBN12(LE34)
        LE36 = ReluActivation(LE35)
        LE37 = ECV7(LE36)
        LE38 = EBN13(LE37)
        LE39 = ReluActivation(LE38)
        
        LE40 = EDW7(LE39)
        LE41 = EBN14(LE40)
        LE42 = ReluActivation(LE41)
        LE43 = ECV8(LE42)
        LE44 = EBN15(LE43)
        LE45 = ReluActivation(LE44)
        
        LE46 = EDW8(LE45)
        LE47 = EBN16(LE46)
        LE48 = ReluActivation(LE47)
        LE49 = ECV9(LE48)
        LE50 = EBN17(LE49)
        LE51 = ReluActivation(LE50)
        
        LE52 = EDW9(LE51)
        LE53 = EBN18(LE52)
        LE54 = ReluActivation(LE53)
        LE55 = ECV10(LE54)
        LE56 = EBN19(LE55)
        LE57 = ReluActivation(LE56)
        
        LE58 = EDW10(LE57)
        LE59 = EBN20(LE58)
        LE60 = ReluActivation(LE59)
        LE61 = ECV11(LE60)
        LE62 = EBN21(LE61)
        LE63 = ReluActivation(LE62)
        
        LE64 = EDW11(LE63)
        LE65 = EBN22(LE64)
        LE66 = ReluActivation(LE65)
        LE67 = ECV12(LE66)
        LE68 = EBN23(LE67)
        LE69 = ReluActivation(LE68)
        
        LE70 = EDW12(LE69)
        LE71 = EBN24(LE70)
        LE72 = ReluActivation(LE71)
        LE73 = ECV13(LE72)
        LE74 = EBN25(LE73)
        LE75 = ReluActivation(LE74)
        
        LE76 = EDW13(LE75)
        LE77 = EBN26(LE76)
        LE78 = ReluActivation(LE77)
        LE79 = ECV14(LE78)
        LE80 = EBN27(LE79)
        LE81 = ReluActivation(LE80)

        LE82 = EAvePool(LE81)
        LE83 = Flatten()(LE82)
        LE84 = EFC1(LE83)
        LE85 = EFC2(LE84)
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

        RE28 = EDW5(RE27)
        RE29 = EBN10(RE28)
        RE30 = ReluActivation(RE29)
        RE31 = ECV6(RE30)
        RE32 = EBN11(RE31)
        RE33 = ReluActivation(RE32)

        RE34 = EDW6(RE33)
        RE35 = EBN12(RE34)
        RE36 = ReluActivation(RE35)
        RE37 = ECV7(RE36)
        RE38 = EBN13(RE37)
        RE39 = ReluActivation(RE38)
        
        RE40 = EDW7(RE39)
        RE41 = EBN14(RE40)
        RE42 = ReluActivation(RE41)
        RE43 = ECV8(RE42)
        RE44 = EBN15(RE43)
        RE45 = ReluActivation(RE44)
        
        RE46 = EDW8(RE45)
        RE47 = EBN16(RE46)
        RE48 = ReluActivation(RE47)
        RE49 = ECV9(RE48)
        RE50 = EBN17(RE49)
        RE51 = ReluActivation(RE50)
        
        RE52 = EDW9(RE51)
        RE53 = EBN18(RE52)
        RE54 = ReluActivation(RE53)
        RE55 = ECV10(RE54)
        RE56 = EBN19(RE55)
        RE57 = ReluActivation(RE56)
        
        RE58 = EDW10(RE57)
        RE59 = EBN20(RE58)
        RE60 = ReluActivation(RE59)
        RE61 = ECV11(RE60)
        RE62 = EBN21(RE61)
        RE63 = ReluActivation(RE62)
        
        RE64 = EDW11(RE63)
        RE65 = EBN22(RE64)
        RE66 = ReluActivation(RE65)
        RE67 = ECV12(RE66)
        RE68 = EBN23(RE67)
        RE69 = ReluActivation(RE68)
        
        RE70 = EDW12(RE69)
        RE71 = EBN24(RE70)
        RE72 = ReluActivation(RE71)
        RE73 = ECV13(RE72)
        RE74 = EBN25(RE73)
        RE75 = ReluActivation(RE74)
        
        RE76 = EDW13(RE75)
        RE77 = EBN26(RE76)
        RE78 = ReluActivation(RE77)
        RE79 = ECV14(RE78)
        RE80 = EBN27(RE79)
        RE81 = ReluActivation(RE80)

        RE82 = EAvePool(RE81)
        RE83 = Flatten()(RE82)
        RE84 = EFC1(RE83)
        RE85 = EFC2(RE84)


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

        FC34 = FDW6(FC33)
        FC35 = FBN12(FC34)
        FC36 = ReluActivation(FC35)
        FC37 = FCV7(FC36)
        FC38 = FBN13(FC37)
        FC39 = ReluActivation(FC38)
        
        FC40 = FDW7(FC39)
        FC41 = FBN14(FC40)
        FC42 = ReluActivation(FC41)
        FC43 = FCV8(FC42)
        FC44 = FBN15(FC43)
        FC45 = ReluActivation(FC44)
        
        FC46 = FDW8(FC45)
        FC47 = FBN16(FC46)
        FC48 = ReluActivation(FC47)
        FC49 = FCV9(FC48)
        FC50 = FBN17(FC49)
        FC51 = ReluActivation(FC50)
        
        FC52 = FDW9(FC51)
        FC53 = FBN18(FC52)
        FC54 = ReluActivation(FC53)
        FC55 = FCV10(FC54)
        FC56 = FBN19(FC55)
        FC57 = ReluActivation(FC56)
        
        FC58 = FDW10(FC57)
        FC59 = FBN20(FC58)
        FC60 = ReluActivation(FC59)
        FC61 = FCV11(FC60)
        FC62 = FBN21(FC61)
        FC63 = ReluActivation(FC62)
        
        FC64 = FDW11(FC63)
        FC65 = FBN22(FC64)
        FC66 = ReluActivation(FC65)
        FC67 = FCV12(FC66)
        FC68 = FBN23(FC67)
        FC69 = ReluActivation(FC68)
        
        FC70 = FDW12(FC69)
        FC71 = FBN24(FC70)
        FC72 = ReluActivation(FC71)
        FC73 = FCV13(FC72)
        FC74 = FBN25(FC73)
        FC75 = ReluActivation(FC74)
        
        FC76 = FDW13(FC75)
        FC77 = FBN26(FC76)
        FC78 = ReluActivation(FC77)
        FC79 = FCV14(FC78)
        FC80 = FBN27(FC79)
        FC81 = ReluActivation(FC80)

        FC82 = FAvePool(FC81)
        FC83 = Flatten()(FC82)
        FC84 = FFC1(FC83)
        FC85 = FFC2(FC84)

        #Face Grid
        FG1 = FGFC1(faceGridInput)
        FG2 = FGFC2(FG1)
        
        #Combining left & right eye face and faceGrid
        dataLRMerge = Concatenate(axis=1)([LE84,RE84,FC84,FG2])
        dataFc1 = createFullyConnected(128)(dataLRMerge)
        finalOutput = createFullyConnected(2,activation = 'linear')(dataFc1)


        #Return the fully constructed model
        return Model(inputs = [leftEyeInput, rightEyeInput, faceInput, faceGridInput], outputs = finalOutput)
