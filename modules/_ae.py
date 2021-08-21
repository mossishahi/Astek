import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input
import pandas as pd
import random
from keras.callbacks import EarlyStopping

from ._base import BaseModule

class AutoEncoder(BaseModule):
    def __init__(self, input_shape, layers, e_activation = 'relu', d_activation = 'tanh', optimizer = 'adam', loss = 'mse'):
        self.layers = layers
        self.inp = Input(shape=(input_shape,))
        #Encoder
        self.encoder = Sequential()
        self.encoder.add(Dense(layers[0], activation = e_activation, input_shape = (input_shape,)))
        print(layers)
        for i in range(1,len(layers)):
            self.encoder.add(Dense(layers[i], activation = e_activation))
        #Decoder
        self.decoder = Sequential()
        for i in range(1,len(layers)):
            self.decoder.add(Dense(layers[-i], activation = d_activation))
        self.decoder.add(Dense(layers[0], activation = 'relu'))
        self.decoder.add(Dense(input_shape, activation = None))
        ## output
        self.model = Model(inputs = self.encoder.input, outputs = self.decoder(self.encoder.output))
        self.model.compile(optimizer = optimizer, loss = loss)
        self.model.summary()
    def train(self, input_X, output_X):
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 20)
        history = self.model.fit(input_X, output_X, epochs = 120, batch_size = 32, verbose = 1, validation_split = 0.2, callbacks = [es])
        return history

    def get_low_dim(self, X):
        dim_reducer = Model(self.encoder.input, self.encoder.output)
        return dim_reducer.predict(X)
    # def save(self, history, base_dir = os.path.abspath("./outputs/")):
    #     self.model.save(str(base_dir + "/saved_models/" + self.model_name + ".h5"))
    #     with open(base_dir + "/train_history/" + str("h_"+self.model_name)+".pickle", 'wb') as file:
    #         joblib.dump(history.history, file)
