import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, LSTM, GRU, Bidirectional, MaxPool1D, TimeDistributed, RepeatVector, Dense, Attention, Input, Embedding, Dropout, BatchNormalization

def get_output_dim(original_dim):
    if original_dim // 1.3 >= 512:
        return 512
    elif original_dim // 1.3 <= 128:
        return 128
    else:
        return int(original_dim // 1.3)

"""### Models

#### SAE
"""

def SAE(n_steps, n_features, activation):   
    encoder = keras.models.Sequential([
        keras.layers.Reshape([n_steps * n_features], input_shape=[n_steps, n_features]),
        Dense(n_features // 1.1, activation=activation),
        Dropout(0.1),
        Dense(n_features // 1.2, activation=activation),
        Dropout(0.1),
        Dense(get_output_dim(n_steps * n_features), kernel_initializer='lecun_normal', activation='selu')
    ])

    decoder = keras.models.Sequential([
        Dense(n_features // 1.2, activation=activation, input_shape=[get_output_dim(n_steps * n_features)]),
        Dropout(0.1),
        Dense(n_features // 1.1, activation=activation),
        Dropout(0.1),
        Dense(n_steps * n_features),
        keras.layers.Reshape([n_steps, n_features])
    ])

    return encoder, decoder

"""#### CNN AE"""

from tensorflow.keras.layers import Conv2DTranspose

def CNN_AE(n_steps, n_features, activation):  
    encoder = keras.models.Sequential([
        Conv1D(32, kernel_size=1, padding="SAME", activation=activation, input_shape=[n_steps, n_features]),
        MaxPool1D(pool_size=2),
        Conv1D(64, kernel_size=3, padding="SAME", activation=activation),
        MaxPool1D(pool_size=2),
        Conv1D(128, kernel_size=5, padding="SAME", activation=activation),
        MaxPool1D(pool_size=2),
        keras.layers.Flatten(),
        Dense(get_output_dim(n_steps * n_features), kernel_initializer='lecun_normal', activation='selu')
    ])

    decoder = keras.models.Sequential([
        keras.layers.Reshape([get_output_dim(n_steps * n_features), 1, 1], input_shape=[get_output_dim(n_steps * n_features)]),
        Conv2DTranspose(filters=64, kernel_size=5, activation=activation),
        Conv2DTranspose(filters=32, kernel_size=3, activation=activation),
        Conv2DTranspose(filters=1, kernel_size=1, activation=activation),
        keras.layers.Flatten(),
        Dense(n_steps * n_features),
        keras.layers.Reshape([n_steps, n_features])
    ])

    return encoder, decoder

"""#### RNN (LSTM / GRU / BiLSTM) AE"""

def LSTM_AE(n_steps, n_features, activation): 
    encoder = keras.models.Sequential([
        LSTM(256, return_sequences=True, input_shape=[n_steps, n_features]),
        Dropout(0.1),
        LSTM(128),
        Dropout(0.1),
        Dense(get_output_dim(n_steps * n_features), kernel_initializer='lecun_normal',  activation='selu')
    ])

    decoder = keras.models.Sequential([
        RepeatVector(n_steps, input_shape=[get_output_dim(n_steps * n_features)]),
        LSTM(128, return_sequences=True),
        Dropout(0.1),
        LSTM(256, return_sequences=True),
        Dropout(0.1),
        TimeDistributed(Dense(n_features))
    ])

    return encoder, decoder

def GRU_AE(n_steps, n_features, activation): 
    encoder = keras.models.Sequential([
        GRU(256, return_sequences=True, input_shape=[n_steps, n_features]),
        Dropout(0.1),
        GRU(128),
        Dropout(0.1),
        Dense(get_output_dim(n_steps * n_features), kernel_initializer='lecun_normal',  activation='selu')
    ])

    decoder = keras.models.Sequential([
        RepeatVector(n_steps, input_shape=[get_output_dim(n_steps * n_features)]),
        GRU(128, return_sequences=True),
        Dropout(0.1),
        GRU(256, return_sequences=True),
        Dropout(0.1),
        TimeDistributed(Dense(n_features))
    ])

    return encoder, decoder

def Bi_LSTM_AE(n_steps, n_features, activation): 
    encoder = keras.models.Sequential([
        Bidirectional(LSTM(256, return_sequences=True, input_shape=[n_steps, n_features])),
        Dropout(0.1),
        Bidirectional(LSTM(128)),
        Dropout(0.1),
        Dense(get_output_dim(n_steps * n_features), kernel_initializer='lecun_normal',  activation='selu')
    ])

    decoder = keras.models.Sequential([
        RepeatVector(n_steps, input_shape=[get_output_dim(n_steps * n_features)]),
        LSTM(128, return_sequences=True),
        Dropout(0.1),
        LSTM(256, return_sequences=True),
        Dropout(0.1),
        TimeDistributed(Dense(n_features))
    ])

    return encoder, decoder

"""#### CNN x BiLSTM AE"""

from tensorflow.keras.layers import Conv2DTranspose

def CNN_Bi_LSTM_AE(n_steps, n_features, activation): 
    encoder = keras.models.Sequential([
        Conv1D(32, kernel_size=1, padding="SAME", activation=activation, input_shape=[n_steps, n_features]),
        MaxPool1D(pool_size=2),
        Conv1D(64, kernel_size=3, padding="SAME", activation=activation),
        MaxPool1D(pool_size=2),
        Conv1D(128, kernel_size=5, padding="SAME", activation=activation),
        MaxPool1D(pool_size=2),
        Bidirectional(LSTM(128)),
        Dense(get_output_dim(n_steps * n_features), kernel_initializer='lecun_normal',  activation='selu')
    ])

    decoder = keras.models.Sequential([
#         RepeatVector(n_steps, input_shape=[get_output_dim(n_steps * n_features)]),
#         LSTM(16, return_sequences=True), # LSTM is significant to accuracy!!!
        keras.layers.Reshape([get_output_dim(n_steps * n_features), 1, 1], input_shape=[get_output_dim(n_steps * n_features)]),
        Conv2DTranspose(filters=32, kernel_size=3, activation=activation),
        Conv2DTranspose(filters=16, kernel_size=1, activation=activation),
        keras.layers.Flatten(),
#         Dense((n_steps * n_features) / 1.2, activation=activation),
        Dense(n_steps * n_features),
        keras.layers.Reshape([n_steps, n_features])
    ])

    return encoder, decoder

def Causal_CNN_AE(n_steps, n_features, activation): 
    encoder = keras.models.Sequential()
    for rate in [1, 2, 4, 8, 16, 32, 64]*2:
        encoder.add(Conv1D(filters=32, kernel_size=2, padding="causal", activation=activation, dilation_rate=rate))
        encoder.add(Dropout(0.1))
    encoder.add(keras.layers.Conv1D(filters=16, kernel_size=1, activation=activation))
    encoder.add(keras.layers.MaxPool1D(pool_size=2))
    encoder.add(keras.layers.Flatten())
    encoder.add(Dense(get_output_dim(n_features), kernel_initializer='lecun_normal',  activation='selu'))

    decoder = keras.models.Sequential([
        keras.layers.Reshape([get_output_dim(n_features), 1, 1], input_shape=[get_output_dim(n_features)]),
        Conv2DTranspose(filters=32, kernel_size=3, activation=activation),
        Conv2DTranspose(filters=16, kernel_size=1, activation=activation),
        keras.layers.Flatten(),
        Dense(n_steps * n_features),
        keras.layers.Reshape([n_steps, n_features])
    ])
    
    return encoder, decoder

from tensorflow.keras.layers import Input, Conv1D, Dense, Activation, Dropout, Lambda, Multiply, Add, Concatenate
from tensorflow.keras.models import Model

def Wavenet(n_steps, n_features, activation):
    # convolutional operation parameters
    n_filters = 32
    filter_width = 2
    dilation_rates = [2**i for i in range(9)] * 3

    # define an input history series and pass it through a stack of dilated causal convolution blocks. 
    history_seq = Input(shape=[n_steps, n_features])
    x = history_seq

    skips = []
    for dilation_rate in dilation_rates:
        # preprocessing - equivalent to time-distributed dense
        x = Conv1D(16, 1, padding='same', activation=activation)(x) 

        # filter convolution
        x_f = Conv1D(filters=n_filters, kernel_size=filter_width, padding='causal',dilation_rate=dilation_rate)(x)
        x_f = Dropout(0.1)(x_f)
        # gating convolution
        x_g = Conv1D(filters=n_filters, kernel_size=filter_width, padding='causal',dilation_rate=dilation_rate)(x)
        x_g = Dropout(0.1)(x_g)
        # multiply filter and gating branches
        z = Multiply()([Activation('tanh')(x_f), Activation('sigmoid')(x_g)])

        # postprocessing - equivalent to time-distributed dense
        z = Conv1D(16, 1, padding='same', activation=activation)(z)

        # residual connection
        x = Add()([x, z])    

        # collect skip connections
        skips.append(z)

    # add all skip connection outputs 
    out = Activation(activation)(Add()(skips))

    # final time-distributed dense layers 
    out = Conv1D(16, 1, activation=activation, padding='same')(out)
    out = MaxPool1D(pool_size=2)(out)
    out = keras.layers.Flatten()(out)
    out = Dense(get_output_dim(n_steps * n_features), kernel_initializer='lecun_normal',  activation='selu')(out)
    encoder = Model(inputs=[history_seq], outputs=[out])

    decoder = keras.models.Sequential([
        keras.layers.Reshape([get_output_dim(n_steps * n_features), 1, 1], input_shape=[get_output_dim(n_steps * n_features)]),
        Conv2DTranspose(filters=32, kernel_size=3, activation=activation),
        Conv2DTranspose(filters=16, kernel_size=1, activation=activation),
        keras.layers.Flatten(),
        Dense(n_steps * n_features),
        keras.layers.Reshape([n_steps, n_features])
    ])
    
    return encoder, decoder

"""#### Attention-based Models"""

def Attention_Bi_LSTM_AE(n_steps, n_features, activation):
    en_input = Input(shape=[n_steps, n_features])
    e = Bidirectional(LSTM(256, recurrent_dropout=0.1, dropout=0.1, return_sequences=True, input_shape=[n_steps, n_features]))(en_input)
    e = Bidirectional(LSTM(128, recurrent_dropout=0.1, dropout=0.1))(e)
    e = Attention(use_scale=True)([e, e])
    en_output = Dense(get_output_dim(n_steps * n_features), kernel_initializer='lecun_normal',  activation='selu')(e)
    encoder = keras.models.Model(inputs=[en_input], outputs=[en_output])
    
    de_input = Input(shape=[get_output_dim(n_steps * n_features)])
    d = RepeatVector(n_steps)(de_input)
    d = LSTM(128, return_sequences=True, recurrent_dropout=0.1, dropout=0.1)(d)
    d = LSTM(256, return_sequences=True, recurrent_dropout=0.1, dropout=0.1)(d)
    d = Attention(use_scale=True, causal=True)([d, d])
    de_output = TimeDistributed(Dense(n_features))(d)
    decoder = keras.models.Model(inputs=[de_input], outputs=[de_output])

    return encoder, decoder

from tensorflow.keras.layers import Conv2DTranspose

def Attention_CNN_Bi_LSTM_AE(n_steps, n_features, activation): 
    en_input = Input(shape=[n_steps, n_features])
    e = Conv1D(32, kernel_size=1, padding="SAME", activation=activation)(en_input)
    e = MaxPool1D(pool_size=2)(e)
    e = Conv1D(64, kernel_size=3, padding="SAME", activation=activation)(e)
    e = MaxPool1D(pool_size=2)(e)
    e = Conv1D(128, kernel_size=5, padding="SAME", activation=activation)(e)
    e = MaxPool1D(pool_size=2)(e)
    e = Bidirectional(LSTM(64, recurrent_dropout=0.1, dropout=0.1))(e)
    e = Attention(use_scale=True)([e, e])
    en_output = Dense(get_output_dim(n_steps * n_features), kernel_initializer='lecun_normal',  activation='selu')(e)
    encoder = keras.models.Model(inputs=[en_input], outputs=[en_output])
    
    decoder = keras.models.Sequential([
        RepeatVector(n_steps, input_shape=[get_output_dim(n_steps * n_features)]),
        LSTM(256, return_sequences=True),
        keras.layers.Reshape([n_steps, 256, 1]),
        Conv2DTranspose(filters=16, kernel_size=3, activation=activation),
        Conv2DTranspose(filters=1, kernel_size=3, activation=activation),
        keras.layers.Flatten(),
        Dense(n_steps * n_features),
        keras.layers.Reshape([n_steps, n_features])
    ])

    return encoder, decoder

from tensorflow.keras.layers import Input, Conv1D, Dense, Activation, Dropout, Lambda, Multiply, Add, Concatenate
from tensorflow.keras.models import Model

def Attention_Wavenet(n_steps, n_features, activation):
    # convolutional operation parameters
    n_filters = 32
    filter_width = 2
    dilation_rates = [2**i for i in range(9)] * 3

    # define an input history series and pass it through a stack of dilated causal convolution blocks. 
    history_seq = Input(shape=[n_steps, n_features])
    x = history_seq

    skips = []
    for dilation_rate in dilation_rates:
        # preprocessing - equivalent to time-distributed dense
        x = Conv1D(16, 1, padding='same', activation=activation)(x) 

        # filter convolution
        x_f = Conv1D(filters=n_filters, kernel_size=filter_width, padding='causal',dilation_rate=dilation_rate)(x)
        x_f = Dropout(0.1)(x_f)
        # gating convolution
        x_g = Conv1D(filters=n_filters, kernel_size=filter_width, padding='causal',dilation_rate=dilation_rate)(x)
        x_g = Dropout(0.1)(x_g)
        # multiply filter and gating branches
        z = Multiply()([Activation('tanh')(x_f), Activation('sigmoid')(x_g)])

        # postprocessing - equivalent to time-distributed dense
        z = Conv1D(16, 1, padding='same', activation=activation)(z)
        # residual connection
        x = Add()([x, z])    

        # collect skip connections
        skips.append(z)

    # add all skip connection outputs 
    out = Activation(activation)(Add()(skips))
    out = Attention(use_scale=True, causal=True)([out, out])
    # final time-distributed dense layers 
    out = Conv1D(16, 1, activation=activation, padding='same')(out)
    out = MaxPool1D(pool_size=2)(out)
    out = keras.layers.Flatten()(out)
    out = Dense(get_output_dim(n_steps * n_features), kernel_initializer='lecun_normal',  activation='selu')(out)
    encoder = Model(inputs=[history_seq], outputs=[out])

    decoder = keras.models.Sequential([
        keras.layers.Reshape([get_output_dim(n_steps * n_features), 1, 1], input_shape=[get_output_dim(n_steps * n_features)]),
        Conv2DTranspose(filters=32, kernel_size=3, activation=activation),
        Conv2DTranspose(filters=16, kernel_size=3, activation=activation),
        keras.layers.Flatten(),
        Dense(n_steps * n_features),
        keras.layers.Reshape([n_steps, n_features])
    ])
    
    return encoder, decoder