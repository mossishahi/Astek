import numpy as np
import tensorflow as tf
from keras import backend as K

class FLoss:
    def loss(self, y_true, y_pred):
        self.y_true  = tf.convert_to_tensor(y_true) 
        self.y_pred  = tf.convert_to_tensor(y_pred)
        tf.print("====================")
        tf.print(y_pred)
        tf.print(y_true)
        # tf.print(K.max(self.y_pred, [[0.00000001]]))
        tf.print("====================")
        s = K.mean(((self.y_true - K.log(K.max([self.y_pred, [[0.00000001]]]))) + ((1-self.y_true) - K.log(K.max([1-self.y_pred, [[0.00000001]]])))), axis = -1)
        return s