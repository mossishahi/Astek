import numpy as np
import tensorflow as tf
from keras import backend as K

class FLoss:

    def loss(self, y, y_pred):
        self.y  = tf.convert_to_tensor(y) 
        self.y_pred  = tf.convert_to_tensor(y_pred)
        s = ((self.y - K.log(self.y_pred)) + ((1-self.y) - K.log(1-self.y_pred)))
        return s