import modules
import sys
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import tabulate
import pickle
import modules
import models 
import numpy as np
import pandas as pd
from modules import Dumper
from sklearn import preprocessing
from keras import backend as K
import tensorflow as tf

# #tensorflow Logs
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
# #project-directory
# PROJ_DIRECTORY = os.getcwd()
# #log, clg: console - flg:file
# clg, flg = modules.MyLog().getLogger()

# sim_df = modules.DataLoader(base = PROJ_DIRECTORY).load_data("simulated-data-raw")
# print(sim_df.iloc[:1000, -1].shape)
# sc = preprocessing.MinMaxScaler()
# norm_x = sc.fit_transform(sim_df.iloc[:1000, 2:]) 
# norm_y = sc.fit_transform(sim_df.iloc[:1000, -1])
# #feed data to Model
# X_train = np.array(norm_x.values, dtype = 'float32').reshape(-1, 20, norm_df.shape[1])
# y_train = np.array(norm_y.values, dtype = 'float32').reshape(-1, 1)
 
# model = models.REGRESSION(X_train.shape[1:], n_outputs = y_train.shape[1])
# history = model.train(X_train, y_train, epochs=5)
# model.save(history.history, model.model_name)
# model.visualize(history.history, model.model_name + "_regression_")

y  = [0.2, 0.3, 0.001]
y_pred = [0.004, 0.1, 0.11]

y  = tf.convert_to_tensor(y) 
y_pred  = tf.convert_to_tensor(y_pred)

# s = ((y - K.log(y_pred)) + ((1-y) - K.log(1-y_pred)))
s = y - y_pred
print(K.mean(s, axis = -1))