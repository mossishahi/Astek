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
        self.encoder = Sequential()
        self.encoder.add(Dense(layers[0], activation = e_activation, input_shape = (input_shape,)))
        for i in range(1,len(layers)):
            self.encoder.add(Dense(layers[i], activation = e_activation))
        #Decoder
        self.decoder = Sequential()
        for i in range(1,len(layers)):
            self.decoder.add(Dense(layers[-i], activation = d_activation))
        self.decoder.add(Dense(layers[0], activation = 'relu')
        ## output
        self.model = Model(inputs = encoder.input, outputs = decoder(encoder.output))
        self.model.compile(optimizer = optimizer, loss = loss)