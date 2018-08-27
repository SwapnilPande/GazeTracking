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
        imageInput =Input(shape=(323,323,3,))
        
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
        paddingF3 = createPadding(1)
        convF4 = createConvLayer(384, 3, 1)
        paddingF4 = createPadding(1)
        convF5 = createConvLayer(128, 3, 1)
        maxPoolF3 = createMaxPool()
        

        #Define fully connected layers for face
        fullyConnected1 = createFullyConnected(128)
        fullyConnected2 = createFullyConnected(128)

        #Define fully connected layers for eyes & face & face grid
        fullyConnected3 = createFullyConnected(64)
        fullyConnected4 = createFullyConnected(2, 'linear')


        #FullImage
        dataConvF1 = convF1(imageInput)
        dataMaxPoolF1 = maxPoolF1(dataConvF1)
        dataBNF1= BNF1(dataMaxPoolF1)
        dataPaddingF1 = paddingF1(dataBNF1)
        dataConvF2 = convF2(dataPaddingF1)
        dataMaxPoolF2 = maxPoolF2(dataConvF2)
        dataBNF2 =BNF2(dataMaxPoolF2)
        dataPaddingF2 = paddingF2(dataBNF2)
        dataConvF3 = convF3(dataPaddingF2)
        dataPaddingF3 = paddingF3(dataConvF3)
        dataConvF4 = convF4(dataPaddingF3)
        dataPaddingF4 = paddingF4(dataConvF4)
        dataConvF5 = convF5(dataPaddingF4)
        dataMaxPoolF3 =maxPoolF3(dataConvF5)
        
        #Reshape data to feed into fully connected layer
        faceFinal = Reshape((10368,))(dataMaxPoolF3)
        dataFullyConnected1 = fullyConnected1(faceFinal)
        dataFullyConnected2 = fullyConnected2(dataFullyConnected1)
        dataFullyConnected3 = fullyConnected3(dataFullyConnected2)
        finalOutput = fullyConnected4(dataFullyConnected3)


        #Return the fully constructed model
        return Model(inputs = imageInput, outputs = finalOutput)
