#Import necessary layers for model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Concatenate, Reshape, ZeroPadding2D,Activation,SeparableConv2D,AveragePooling2D,Flatten,DepthwiseConv2D,ReLU,Dropout,Lambda,Add
#Import initializers for weights and biases
from tensorflow.keras.initializers import Zeros, RandomNormal
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import BatchNormalization
import numpy as np
import tensorflow.keras.backend as K
import tensorflow_model_optimization as tfmot 
from tensorflow_model_optimization.sparsity import keras as sparsity
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
def createCv(input, filters, kernelSize, stride,padding='same',prune=False,pruning_schedule=None):
        if prune:
            return sparsity.prune_low_magnitude(Conv2D(
                    filters,
                    kernelSize, 
                    strides = stride,
                    activation = None,
                    use_bias = True,
                    kernel_initializer = randNormKernelInitializer(),
                    bias_initializer = 'zeros',
                    padding=padding
                    ),pruning_schedule=pruning_schedule)(input)
        else:
                
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


def createFullyConnected(input,units, activation = 'relu',prune=False,pruning_schedule=None):
        if prune:
            return sparsity.prune_low_magnitude(Dense(
                units,
                activation = activation,
                use_bias = True,
                kernel_initializer = randNormKernelInitializer(),
                bias_initializer = 'zeros'
                ),pruning_schedule=pruning_schedule)(input)
        else:
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


def _inverted_res_block(inputs, expansion, stride, filters, block_id, alpha=1,name="left",prune=False,pruning_schedule=None):
    in_channels =K.int_shape(inputs)[-1]
    pointwise_filters = int(filters * alpha)
    #pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'block_{}_{}'.format(name,block_id)

    if block_id:
        # Expand
        if prune:
            x=sparsity.prune_low_magnitude(Conv2D(expansion * in_channels,
                          kernel_size=1,
                          padding='same',
                          use_bias=False,
                          activation=None,
                          name=prefix + 'expand'),pruning_schedule=pruning_schedule)(x)
        else:
            x = Conv2D(expansion * in_channels,
                          kernel_size=1,
                          padding='same',
                          use_bias=False,
                          activation=None,
                          name=prefix + 'expand')(x)
        x = BatchNormalization(epsilon=1e-3,
                               momentum=0.999,
                               name=prefix + 'expand_BN')(x)
        x = ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_{}'.format(name)

    # Depthwise
    if prune:
        x=sparsity.prune_low_magnitude(DepthwiseConv2D(kernel_size=3,
                        strides=stride,
                        activation=None,
                        use_bias=False,
                        padding='same',
                        name=prefix + 'depthwise'),pruning_schedule=pruning_schedule)(x)
    else:
        x = DepthwiseConv2D(kernel_size=3,
                        strides=stride,
                        activation=None,
                        use_bias=False,
                        padding='same',
                        name=prefix + 'depthwise')(x)
    x = BatchNormalization(epsilon=1e-3,
                           momentum=0.999,
                           name=prefix + 'depthwise_BN')(x)
    
    x = ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    if prune:
        x=sparsity.prune_low_magnitude(Conv2D(pointwise_filters,
               kernel_size=1,
               padding='same',
               use_bias=False,
               activation=None,
               name=prefix + 'project'),pruning_schedule=pruning_schedule)(x)
    else:
        x = Conv2D(pointwise_filters,
               kernel_size=1,
               padding='same',
               use_bias=False,
               activation=None,
               name=prefix + 'project')(x)
    x = BatchNormalization(
        epsilon=1e-3, momentum=0.999, name=prefix + 'project_BN')(x)

    if in_channels == pointwise_filters and stride == 1:
        return Add(name=prefix + 'add')([inputs, x])
    return x

def createEyeModelV2_1(inputs,name="eye",prune=False,pruning_schedule=None):
        #EyeInput = Input(shape=(224,224,3,))
        ## standard ConV+BN+activation
        x = createCv(inputs,32,5,4,padding='same',prune=prune,pruning_schedule=pruning_schedule)  #3@224x224 -->> 32@112x112
        x = createBN(x)
        x = createActivation(x)
        ## bottleneck block
        x = _inverted_res_block(inputs=x, filters =24 ,stride =1 , expansion =1 , block_id=0,alpha=1,name=name,prune=prune,pruning_schedule=pruning_schedule)      #32@112X112 --> 16@112x112
        
        x = _inverted_res_block(inputs=x, filters =24 ,stride =2 , expansion =4 , block_id=1,alpha=1,name=name,prune=prune,pruning_schedule=pruning_schedule)          #16@112x112 --> 24@56x56
        x = _inverted_res_block(inputs=x, filters =24 ,stride =1 , expansion =4 , block_id=2,alpha=1,name=name,prune=prune,pruning_schedule=pruning_schedule)          #24@56x56 -->24@56x56
        
        x = _inverted_res_block(inputs=x, filters =32 ,stride =2 , expansion =4 , block_id=3,alpha=1,name=name,prune=prune,pruning_schedule=pruning_schedule)           #24@56x56 --> 32@28x28 
        x = _inverted_res_block(inputs=x, filters =32 ,stride =1 , expansion =4 , block_id=4,alpha=1,name=name,prune=prune,pruning_schedule=pruning_schedule)
        #x = _inverted_res_block(inputs=x, filters =32 ,stride =1 , expansion =4 , block_id=5,alpha=1,name=name)

        x = _inverted_res_block(inputs=x, filters =64 ,stride =2 , expansion =4 , block_id=6,alpha=1,name=name,prune=prune,pruning_schedule=pruning_schedule)          #32@28x28-->>64@14x14    
        x = _inverted_res_block(inputs=x, filters =64 ,stride =1 , expansion =4 , block_id=7,alpha=1,name=name,prune=prune,pruning_schedule=pruning_schedule)
        x = _inverted_res_block(inputs=x, filters =64 ,stride =1 , expansion =4 , block_id=8,alpha=1,name=name,prune=prune,pruning_schedule=pruning_schedule)
        #x = _inverted_res_block(inputs=x, filters =64 ,stride =1 , expansion =4 , block_id=9,alpha=1,name=name)
        
        x = _inverted_res_block(inputs=x, filters =96 ,stride =1 , expansion =4 , block_id=10,alpha=1,name=name,prune=prune,pruning_schedule=pruning_schedule)         #64@14x14 --> 96@14x14   
        x = _inverted_res_block(inputs=x, filters =96 ,stride =1 , expansion =4 , block_id=11,alpha=1,name=name,prune=prune,pruning_schedule=pruning_schedule)
        #x = _inverted_res_block(inputs=x, filters =96 ,stride =1 , expansion =4 , block_id=12,alpha=1,name=name)
        
        x = _inverted_res_block(inputs=x, filters =160 ,stride =2 , expansion =4 , block_id=13,alpha=1,name=name,prune=prune,pruning_schedule=pruning_schedule)        #96@14x14   --> 160@7x7
        x = _inverted_res_block(inputs=x, filters =160 ,stride =1 , expansion =4 , block_id=14,alpha=1,name=name,prune=prune,pruning_schedule=pruning_schedule)
        #x = _inverted_res_block(inputs=x, filters =160 ,stride =1 , expansion =4 , block_id=15,alpha=1,name=name)

        x = _inverted_res_block(inputs=x, filters =320 ,stride =1 , expansion =4 , block_id=16,alpha=1,name=name,prune=prune,pruning_schedule=pruning_schedule)        #160@7x7 -->>320@7x7

        x= createCv(x,1280,1,1,padding='same',prune=prune,pruning_schedule=pruning_schedule)                                                                 #1280@7x7
        x= createBN(x)
        x= ReLU6(x)

        x=createMaxPool(x,size=4)

        x=Flatten()(x)
        return x
        #return Model(inputs=EyeInput,outputs = x)
def createFaceModelV2_1(inputs,prune=False,pruning_schedule=None):

        x =  createEyeModelV2_1(inputs=inputs,name="face",prune=prune,pruning_schedule=pruning_schedule)
        #x = faceModel(inputs)
        x = createFullyConnected(x,128,prune=prune,pruning_schedule=pruning_schedule)
        x = Dropout(0.5)(x)
        x = createFullyConnected(x,64,prune=prune,pruning_schedule=pruning_schedule)
        x = Dropout(0.5)(x)

        return x
        


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
        F11 = Dropout(0.5)(F11)
        F12 = createFullyConnected(F11,64)
        F12 = Dropout(0.5)(F12)
        
        return F12

def createFaceGridModel(input):
        FG1 = createFullyConnected(input, 256)
        FG2 = createFullyConnected(FG1, 256)
        return FG2

def createEyeLocationModel(input):
        EL1 = createFullyConnected(input,512)
        EL2 = createFullyConnected(EL1, 256)
        return EL2
def createMarkerModel(input,prune=False,pruning_schedule=None):
        MM0 = createFullyConnected(input,128,prune=prune,pruning_schedule=pruning_schedule)
        MM1 = createFullyConnected(MM0,256,prune=prune,pruning_schedule=pruning_schedule)
        MM1 = Dropout(0.5)(MM1)
        MM2 = createFullyConnected(MM1,512,prune=prune,pruning_schedule=pruning_schedule)
        MM2 = Dropout(0.5)(MM2)
        MM3 = createFullyConnected(MM2,256,prune=prune,pruning_schedule=pruning_schedule)
        MM3 = Dropout(0.5)(MM3)
        MM4 = createFullyConnected(MM3,128,prune=prune,pruning_schedule=pruning_schedule)
        MM4 = Dropout(0.5)(MM4)
        return MM4

def initializeModel():   #mobile net v2_0
        
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



def initializeModel_V2(prune=False,pruning_schedule=None):  #mobile net V2_1 use residual blocks
        
        print("Initializing Model")
        #Defining input here
        leftEyeInput = Input(shape=(224,224,3,))
        rightEyeInput = Input(shape=(224,224,3,))
        faceInput = Input(shape=(224,224,3,))
#        faceGridInput = Input(shape=(6400,))
#        EyeLocationInput = Input(shape=(8,))
        MarkerInput = Input(shape=(10,))
        #EyeModel = createEyeModelV2_1(prune=prune,pruning_schedule=pruning_schedule)
        ### eye models
        leftEyeData = createEyeModelV2_1(leftEyeInput,name='leftEye',prune=prune,pruning_schedule=pruning_schedule)
        rightEyeData= createEyeModelV2_1(rightEyeInput,name='rightEye',prune=prune,pruning_schedule=pruning_schedule)
        faceData = createFaceModelV2_1(faceInput,prune=prune,pruning_schedule=pruning_schedule)
#        faceGridData=createFaceGridModel(faceGridInput)
        markerData = createMarkerModel(MarkerInput,prune=prune,pruning_schedule=pruning_schedule)
        
        EyeMerge =  Concatenate(axis=1)([leftEyeData,rightEyeData])
        EyeFc1  = createFullyConnected(EyeMerge,128,prune=prune,pruning_schedule=pruning_schedule)
        EyeFc1  = Dropout(0.5)(EyeFc1)

        
        #Combining left & right eye face and faceGrid
        #dataLRMerge = Concatenate(axis=1)([EyeFc1,markerData])
        dataLRMerge = Concatenate(axis=1)([EyeFc1,faceData,markerData])
        dataFc1 = createFullyConnected(dataLRMerge,128,prune=prune,pruning_schedule=pruning_schedule)
        dataFc1 = Dropout(0.5)(dataFc1)
        finalOutput = createFullyConnected(dataFc1,2,activation = 'linear',prune=prune,pruning_schedule=pruning_schedule)


        #Return the fully constructed model
        #return Model(inputs = [leftEyeInput, rightEyeInput,  MarkerInput], outputs = finalOutput)
        return Model(inputs = [leftEyeInput, rightEyeInput, faceInput, MarkerInput], outputs = finalOutput)

def initializeModel_V1():
        
        print("Initializing Model")
        #Defining input here
        leftEyeInput = Input(shape=(224,224,3,))
        rightEyeInput = Input(shape=(224,224,3,))
        faceInput = Input(shape=(224,224,3,))
#        faceGridInput = Input(shape=(6400,))
#        EyeLocationInput = Input(shape=(8,))
        MarkerInput = Input(shape=(10,))
        
        ### eye models
        leftEyeData = createEyeModel(leftEyeInput)
        rightEyeData= createEyeModel(rightEyeInput)
        faceData = createFaceModel(faceInput)
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

def myFun(x):
        return (K.sin(x[0]-1.05)/K.cos(x[0]-1.05))*(x[1]+5.0)+x[2]

def myCos(x):
        return K.cos(x[0]+x[1])

def initializeModel_lambda():


        print("Initializing Model")
        # load eye and face model from pretrained model
        eye_model = load_model('Eye_model.hdf5')
        face_model=load_model('face_model.hdf5')
        # freez the weights
        for l in eye_model.layers:
                l.trainable=False
        for l in face_model.layers:
                l.trainable=False
        
        #Defining input here
        leftEyeInput = Input(shape=(224,224,3,))
        rightEyeInput = Input(shape=(224,224,3,))
        faceInput = Input(shape=(224,224,3,))
#        faceGridInput = Input(shape=(6400,))
#        EyeLocationInput = Input(shape=(8,))
        MarkerInput = Input(shape=(10,))
        
        ### eye models
        leftEyeData = eye_model(leftEyeInput)
        rightEyeData= eye_model(rightEyeInput)
        faceData = face_model(faceInput)
        faceData = createFullyConnected(faceData,128)
        faceData = Dropout(0.5)(faceData)
        faceData = createFullyConnected(faceData,64)
        faceData = Dropout(0.5)(faceData)

#        faceGridData=createFaceGridModel(faceGridInput)
        markerData = createMarkerModel(MarkerInput)
        
        EyeMerge =  Concatenate(axis=1)([leftEyeData,rightEyeData])
        EyeFc1  = createFullyConnected(EyeMerge,128)
        #EyeFc1  = Dropout(0.5)(EyeFc1)
        #EyePose = createFullyConnected(EyeFc1,2)#,activation = 'linear')

        EyeFaceMerge =  Concatenate(axis=1)([EyeFc1,faceData])
        EyeFaceFc1  = createFullyConnected(EyeFaceMerge,128)
        
        EyeGaze = createFullyConnected(EyeFaceFc1,2,activation = 'linear')
        EyeGaze = ReLU(threshold = 0.0,max_value =2.1 )(EyeGaze)  #-80 deg to 80 deg 
        Bias     = createFullyConnected(markerData,2,activation = 'linear')
        Distance = createFullyConnected(markerData,1,activation = 'linear')
        Distance = ReLU(threshold = 0.0,max_value = 200.0)(Distance)
        
        #Combining left & right eye face and faceGrid
        #dataLRMerge = Concatenate(axis=1)([EyeFc1,markerData])
        #cosOutput = Lambda(myCos)([EyePose,FacePose])
        #cosRelu   = ReLU(threshold=0.01)(cosOutput)
                                  
        
        finalOutput = Lambda(myFun)([EyeGaze,Distance,Bias])


        #Return the fully constructed model
        #return Model(inputs = [leftEyeInput, rightEyeInput,  MarkerInput], outputs = finalOutput)
        return Model(inputs = [leftEyeInput, rightEyeInput, faceInput, MarkerInput], outputs = finalOutput)
