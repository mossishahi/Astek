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

import os

#os.environ["CUDA_VISIBLE_DEVICES"]="-1"    

import tensorflow as tf

#tensorflow Logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
#project-directory
PROJ_DIRECTORY = os.getcwd()
#log, clg: console - flg:file
clg, flg = modules.MyLog().getLogger()
#----------------------------------------------------------------------------------

"""
Experiment 3:
-------------


"""
portion = 0.5
TEST_PORTION = 0.2

# Loading Data
sim_df = modules.DataLoader(base = PROJ_DIRECTORY).load_data("simulated-data-raw")

if "WEEK_DAY" not in sim_df.columns:
    sim_df.insert(7, "WEEK_DAY", sim_df["TX_DATETIME"].apply(lambda x : x.weekday()))

#Feature Selection
selected_features = ["CUSTOMER_ID", "TX_TIME_SECONDS", 'TX_AMOUNT']

#Preprocess Data
pre_proc = modules.Preprocessor(sim_df, portion, [clg, flg])
input_tensors, message = pre_proc.pre_process(selected_features, ['TX_AMOUNT'],
                    numericals = ["TX_AMOUNT", "TX_TIME_SECONDS"],
                    categoricals = None,
                    window_size = 32,
                    drop_rollbase=True,
                    roll_base = ["CUSTOMER_ID", "TX_TIME_SECONDS"])

if input_tensors:
        #---------- Test and Train split ------------
        input_x, y_tensor = input_tensors
        clg.info(str(input_x.shape)+str(y_tensor.shape))
        # clg.info(str(input_x))
        # clg.info("pca comps >> " + str(int(0.1 * input_x.shape[2])))
        clg.info(">>"+str(input_x[:, :, :-1].shape))
        
        train_idx = np.random.choice(input_x.shape[0], 
                                      int(input_x.shape[0]*(1-TEST_PORTION)), replace=False)                      
        X_train = input_x[train_idx, :, :]
        X_test = np.delete(input_x, train_idx, axis=0)
        y_train = y_tensor[train_idx, ]
        y_test = np.delete(y_tensor, train_idx, axis=0)

        input_tensors = [X_train, X_test, y_train, y_test]
        Dumper(PROJ_DIRECTORY + "/data/tensors/train_tensors/").dump(input_tensors, 
                               pre_proc.tensor_name,
                               "X_train, X_test, y_train, y_test".replace(",","").replace("", "").split(" "))

        clg.info(str(X_train.shape) + str(X_test.shape) + str(y_train.shape) + str(y_test.shape))
else:
    clg.warning(message)
    flg.warning(message)
    raise Exception('input Data is empty')
sc = preprocessing.MinMaxScaler()
norm_x = sc.fit_transform(sim_df.iloc[:1000, 2:]) 
norm_y = sc.fit_transform(sim_df.iloc[:1000, 4].to_frame())
# tf.print("Norm Y")
# tf.print(norm_y)
#feed data to Model
X_train = np.array(norm_x, dtype = 'float32').reshape(-1, 20, norm_x.shape[1])
y_train = np.array(norm_y[:50], dtype = 'float32')
#print("X shape")
#print(X_train.shape)
#print(X_train)
#print("=================")
tf.print("y shape")
tf.print(y_train.shape)
#tf.print(y_train)
tf.print("------------------------")
model = models.REGRESSION(X_train.shape[1:], n_outputs = y_train.shape[1])
history = model.train(X_train, y_train, epochs=150)
model.save(history.history, model.model_name)
model.visualize(history.history, model.model_name + "_regression_")
print("32")