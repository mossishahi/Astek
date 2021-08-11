import numpy as np
class FLoss:
    def __init__(self, labels , y_pred):
        self.labels  = labels 
        self.y_pred  = y_pred

    def loss(self):
        s = ((self.labels - np.log2(self.y_pred)) + ((1-self.labels) - np.log2(1-self.y_pred)))
        return s