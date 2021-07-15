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
from keras.callbacks import CSVLogger
from modules import SIMPLE_LSTM

class REGRESSION:
    def __init__(self,
                 input_shape,
                 loss = 'mse', 
                 drop_out = 0.2,
                 metrics = 'mae',
                 n_outputs = 1,
                 optimizer = 'adam'):
        print("inpshape", input_shape)
        lstm_layers = [input_shape[0]]
        self.model = SIMPLE_LSTM(input_shape,
                                lstm_layers = [input_shape[0]],
                                loss = loss, 
                                drop_out=drop_out,
                                metrics = metrics).model
                                        #model's name
        self.model_name = "L" + "".join(map(str,lstm_layers)) + "-"
        self.model_name += "W" + str(input_shape[1])
        self.model_name += ">>" + str(n_outputs)

    def train(self, train_x, train_y, batch_size = 100, epochs = 2, validation_split = 0.2):
        csv_logger = CSVLogger(os.path.abspath("./logs/train.log"), append=True, separator=';')
        print("train shapes:", train_x.shape, train_y.shape)
        result = self.model.fit(train_x,
                                train_y, 
                                batch_size = batch_size,
                                validation_split = validation_split,
                                epochs = epochs,
                                callbacks=[csv_logger])
        return result
    def predict(self, test_x):
        return self.model.predict(test_x)
    def save(self, history, base_dir = os.path.abspath("./outputs/")):
        self.model.save(str(base_dir + "/saved_models/" + self.model_name + ".h5"))
        with open(base_dir + "/train_history/" + str("h_"+self.model_name)+".pickle", 'wb') as file:
            joblib.dump(history.history, file)
