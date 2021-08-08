import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input
import pandas as pd
import random

class AutoEncoder:
    def __init__(self, input_shape, layers, e_activation = 'relu', d_activation = 'tanh', optimizer = 'adam', loss = 'mse'):
        self.layers = layers
        inp = Input(shape=(input_shape,))
        print(layers, type(layers))
        #Encoder
        x = Dense(layers[0], input_shape = (input_shape,), activation = e_activation)(inp)
        for i in range(1,len(layers)):
            x = Dense(layers[i], activation = e_activation)(x)
        #Decoder
        for i in range(1,len(layers)):
            x = Dense(layers[-i], activation = d_activation)(x)
        x = Dense(layers[0])(x)
        ## output
        out = Dense(input_shape, activation = 'relu')(x)
        self.model = Model(inp, out)
        self.model.compile(optimizer = optimizer, loss = loss)