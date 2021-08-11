import numpy as np
import tensorflow as tf
from keras import backend as K

class FLoss:
    def loss(self, y_true, y_pred):
        self.y_true  = tf.convert_to_tensor(y_true) 
        self.y_pred  = tf.convert_to_tensor(y_pred)
        s = K.mean(((self.y_true - K.log(self.y_pred)) + ((1-self.y_true) - K.log(1-self.y_pred))), axis = -1)
        return s