import _loss as ls
import numpy as np
from sklearn.metrics import mean_squared_error
y = np.array([0.4])
y_pred = np.array([0.1])
f=ls.FLoss(y, y_pred)
s=f.loss()
print(s)