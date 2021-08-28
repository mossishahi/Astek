import os
import sys
sys.path.append(os.path.abspath("../../projects/"))
import joblib
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import CSVLogger, EarlyStopping
from modules import SIMPLE_LSTM
from modules import VC

import matplotlib.pyplot as plt
from ._base import BaseModel


class REGRESSION(BaseModel):
    def __init__(self,
                 input_shape,
                 loss = 'mse', 
                 drop_out = 0.2,
                 metrics = 'mae',
                 n_outputs = 1,
                 optimizer = 'adam'):

        lstm_layers = [input_shape[0]]
        self.model = SIMPLE_LSTM(input_shape,
                                lstm_layers = [input_shape[0]],
                                loss = loss, 
                                drop_out=drop_out,
                                metrics = metrics).model
        version = VC().read_version()
        self.model_name = "V:" +  str(version) + "-"
        print("version", version)
<<<<<<< HEAD
        self.model_name = self.model_name + "LSTM:" + "".join(map(str,lstm_layers)) + "-" #number of LSTM Units
        self.model_name += "X" + str(input_shape[1]) #number of Features
        self.model_name += "Y" + str(n_outputs)

    def train(self, train_x, train_y, batch_size = 64, epochs = 150, validation_split = 0.2):
        print("epochs", epochs)
        csv_logger = CSVLogger(os.path.abspath("./logs/train.log"), append=True, separator=';')
        es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=40)
=======
        self.model_name = "LSTM:" + "".join(map(str,lstm_layers)) + "-" #number of LSTM Units
        self.model_name += "X" + str(input_shape[1]) #number of Features
        self.model_name += "Y" + str(n_outputs)

    def train(self, train_x, train_y, batch_size = 64, epochs = 100, validation_split = 0.2):
        print("epochs", epochs)
        csv_logger = CSVLogger(os.path.abspath("./logs/train.log"), append=True, separator=';')
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=40)
>>>>>>> 07e1521aac124893b6ece4cb1206eddd278b9e5e

        print("train shapes:", train_x.shape, train_y.shape)
        result = self.model.fit(train_x,
                                train_y, 
                                batch_size = batch_size,
                                validation_split = validation_split,
                                epochs = epochs,
<<<<<<< HEAD
                                callbacks=[csv_logger])
=======
                                callbacks=[csv_logger, es])
>>>>>>> 07e1521aac124893b6ece4cb1206eddd278b9e5e
        return result
    def predict(self, test_x):
        return self.model.predict(test_x)
    

