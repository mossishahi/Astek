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
#tensorflow Logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
#project-directory
PROJ_DIRECTORY = os.getcwd()
#log, clg: console - flg:file
clg, flg = modules.MyLog().getLogger()
#----------------------------------------------------------------------------------

"""

Experiment 4:
-------------
New Dataset: Kaggle
Univariate Time series prediction based on 
"V1 : V28" + "Time"
"""

# Loading Data
sim_df = modules.DataLoader(base = PROJ_DIRECTORY).load_csv("kaggle-creditcard")

#Feature Selection
pca_components = ["V" + str(i) for i in range(1, 28)]

selected_features = pca_components + ["Time", "Amount"]  #CUSTOMER_ID is removed after making windows

#Preprocess Data
portion = 1
pre_proc = modules.Preprocessor(sim_df, portion, [clg, flg])
input_tensors, message = pre_proc.pre_process(selected_features, ['Amount'],
                    numericals = ["Amount", "Time"],
                    window_size = 64,
                    drop_rollbase=True,
                    roll_base = ["Time"])

if input_tensors:
        #---------- Test and Train split ------------
        input_x, y_tensor = input_tensors
        clg.info(str(input_x.shape)+str(y_tensor.shape))
        clg.info(">>"+str(input_x[:, :, :-1].shape))

        # train<>test split        
        TEST_PORTION = 0.2
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

#feed data to Model
model = models.REGRESSION(X_train.shape[1:], n_outputs = y_train.shape[1])
history = model.train(X_train, y_train, epochs=150)
model.save(history, model.model_name + "_kaggle_")
