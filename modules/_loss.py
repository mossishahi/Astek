import numpy as np
import tensorflow as tf
from keras import backend as K

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession



class FLoss:
    def loss(self, y_true, y_pred):
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        self.y_true  = tf.convert_to_tensor(y_true) 
        self.y_pred  = tf.convert_to_tensor(y_pred)
        tf.print("=========== shapes =========")
        tf.print(y_pred.shape)
        tf.print(y_true.shape)
        # tf.print(K.max(self.y_pred, [[0.00000001]]))
        tf.print("====================")
        # tf.print(K.log(K.max([self.y_pred, [[0.00000001]]])))
        # tf.print(K.log(K.max([1-self.y_pred, [[0.00000001]]])))
        tf.print("--------------------------")
        s = K.mean(((self.y_true - K.log(self.y_pred + 0.000001)) + ((1-self.y_true) - K.log(1-self.y_pred + 0.000001))), axis = -1)
        return s