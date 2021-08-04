import numpy as np
class Loss:
    def __init__(self, labels , y_pred):
        self.labels  = labels 
        self.y_pred  = y_pred

    def cal (self):
        loss = (self.labels - np.log(self.y_pred)) + ((1-self.labels) - np.log(1-self.y_pred))).sum()
        return loss