import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import random

class AutoEncoder:
    def __init__(self, layers, input_shape, e_activation = 'relu', d_activation = 'tanh'):
        self.layers = layers
        inp = Input(shape=(input_shape,))
        encoder = Sequential()
        decoder = Sequential()
        encoder.add(Dense(layers[0], activation = e_activation, input_shape = input_shape))
        
        for i in in range(1,len(layers)):
            encoder.add(Dense(layers[i], activation = e_activation))
            decoder.add(Dense(layers[-i], activation = d_activation))
        
        decoder.add(Dense(layers[0]))
        ## output
        out = Dense(input_shape, activation='relu')(decoder)
        self.model = Model(inp, out)