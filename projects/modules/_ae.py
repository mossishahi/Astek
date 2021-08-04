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
        for i in in len(layers):
            encoder.add(Dense(layers[i], activation = e_activation))
            decoder.add(Dense(layers[-(i+1)], activation = d_activation))
        ## output
        output = Dense(input_shape, activation='relu')(x)
