"""
The following script contains the  Attention based ResNet50
"""
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
import tensorflow as tf
import keras
from keras import layers
import keras.backend as K
    
def AttentionModule(x, gating):
    '''
    The following function implements Self-Attention to a given input x
    Parameters:
    x: input vetor, it will work as our Key
    gating : gating output of x, it will work as our Query
    '''
    #shape_x = K.int_shape(x)
    #shape_g = K.int_shape(gating)

    
    # We will then adjust t shape, so it has the same as x
    #theta_g = layers.Conv2D(shape_x[-1], (1,1), strides=(2,2), padding='same')(gating)
    upsample_g = layers.Conv2DTranspose(x.shape[-1], (1, 1), padding='same')(gating) 

    # We will calculate the attention score by adding x and gating
    concat_xg = layers.add([upsample_g, x])
    act_xg = layers.Activation('relu')(concat_xg)

    # We will convert attention scores into attetion weights
    # To convert them into probabilities we will apply sigomoid function
    # After that we will scale them into x's shape
    weights = layers.Conv2D(1, (1, 1), padding='same')(act_xg)
    weights = layers.Activation('sigmoid')(weights)

    # Once we have our weights, we can multipy them witht the input vector
    y = layers.multiply([weights, x])

    # Finally we will return as an output the desired filters
    #result = layers.Conv2D(shape_x[3], (1, 1), padding='same')(y)
    #result_bn = layers.BatchNormalization()(result)
    return y

def ResidualModule(model, input, output):
    '''
    TODO COMENTAR
    '''
    if input:
        input_layer = layers.Input(model.get_layer(input).output.shape[1:]) 
        # Create the new nodes for each layer in the path
        x = input_layer
        add_layer = input_layer
        i_layer = model.get_layer(input)
        o_layer = model.get_layer(output)
        conv_block_layers = [model.get_layer('conv2_block1_0_conv'),\
                             model.get_layer('conv3_block1_0_conv')]
        count = 0
        flag = False
        for layer in model.layers:
            if layer == i_layer:
                flag = True
            if flag:
                try:
                    if layer not in conv_block_layers:
                        x = layer(x)
                        x.trainable=False
                    else:
                        add_layer = layer(add_layer)
                        add_layer.trainable=False
                    if layer == o_layer: 
                        break
                except:
                    try:
                        add_layer = AttentionModule(x, add_layer)
                        x = layer([x, add_layer]) 
                        x.trainable=False
                        add_layer = x
                        if layer == o_layer: 
                            break
                    except:
                        add_layer = layers.MaxPooling2D()(add_layer)
                        add_layer = layers.MaxPooling2D()(add_layer)
                        x = layer([x, add_layer]) 
                        x.trainable=False
                        add_layer = x
                        if layer == o_layer: 
                            break

        return keras.Model(inputs = input_layer, outputs = x)
    else:
        return keras.Model(inputs = model.input, outputs = model.get_layer(output).output)
    
def get_ResAttLSTMNet(input_shape, seq = 300, dropout_rate=0.5,\
                      type_layers='lstm', num_layers=1, num_units=256):
    '''
    This functions returns a costumized version on Attention ResNet50 
    '''
    # First we will load a pretrained ResNet and freeze it's weight
    resnet =  ResNet50V2(include_top=False, weights='imagenet', pooling='avg',\
                         input_shape = input_shape)
    resnet.trainable = False  

    # We will then define our architecture
   
    ''' ENCODING MODULE '''
    # We will implement a Soft Attention after after every convolutional residual modul
    input = layers.Input(input_shape, dtype=tf.float32)
    
    ## 1ST CONVOLUTION
    resblock1 = ResidualModule(resnet, '', 'pool1_pool') # (56,56,64)
    conv1 = resblock1(input)
    ## 2ND CONCOLUTION
    # 1st block 
    resblock2 = ResidualModule(resnet, 'conv2_block1_preact_bn', 'conv2_block1_out') # (56,56,256)
    conv2 = resblock2(conv1)
    # 2nd block 
    resblock3 = ResidualModule(resnet, 'conv2_block2_preact_bn', 'conv2_block2_out') # (56,56,256)
    conv3 = resblock3(conv2)
    # 3rd block 
    resblock4 = ResidualModule(resnet, 'conv2_block3_preact_bn', 'conv2_block3_out') # (14,14,256)
    conv4 = resblock4(conv3)
    ## 3RD CONVOLUTION
    # 1st block 
    #resblock5 = ResidualModule(resnet, 'conv3_block1_preact_bn', 'conv3_block1_out') # (28,28,512)
    #conv5 = resblock5(conv4)

    # Global Pooling 
    avg_pooling= keras.layers.GlobalAveragePooling2D()(conv4)
    
    ''' DECODING MODULE '''
    # For the decoding module we will define a LSTM network based on the given parameters
    dropout =  layers.Dropout(dropout_rate)(avg_pooling)
    dropout = layers.Reshape((1,256))(dropout)

    if type_layers == 'lstm':
        lstm = layers.LSTM(num_units, return_sequences=True) (dropout[None])
        for i in range(num_layers - 1):
            num_units = num_units // 2
            lstm = layers.LSTM(num_units, return_sequences=True)(lstm)
    elif type_layers == 'bilstm':
        lstm = layers.Bidirectional(layers.LSTM(num_units, return_sequences=True)) (dropout)
        for i in range(num_layers - 1):
            num_units = num_units // 2
            lstm = layers.Bidirectional(layers.LSTM(num_units, return_sequences=True))(lstm)
    else:
         print('ValueError: Layer not supported.')
         return

    output = layers.TimeDistributed(layers.Dense(1, activation='sigmoid'))(lstm)

    model = keras.models.Model(input, output, name="AttentionResLSTMNet")
    return model