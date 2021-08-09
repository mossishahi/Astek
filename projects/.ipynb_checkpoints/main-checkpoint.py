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
TEST_PORTION = 0.2

# Loading Data
sim_df = modules.DataLoader(base = PROJ_DIRECTORY).load_data("simulated-data-raw")

#Feature Define
if "WEEK_DAY" not in sim_df.columns:
    sim_df.insert(7, "WEEK_DAY", sim_df["TX_DATETIME"].apply(lambda x : x.weekday()))
#Feature Selection
selected_features = ["CUSTOMER_ID",
                     "TERMINAL_ID",
                     'TX_TIME_SECONDS', 
                     "WEEK_DAY",
                    'TX_AMOUNT']
                     
flg.info('selected Features are:' + str(selected_features))

# calc = modules.Calculator(sim_df, 8).get_portion_size()

#Preprocess Data
portion = 0.055
pre_proc = modules.Preprocessor(sim_df, portion, [clg, flg])
input_tensors, message = pre_proc.pre_process(selected_features, ['TX_AMOUNT'],
                    numericals = ["TX_AMOUNT", "TX_TIME_SECONDS",'TX_TIME_DAYS'],
                    categoricals = ["TERMINAL_ID", "WEEK_DAY"], 
                    window_size = 64,
                    drop_rollbase=True,
                    roll_base = ["CUSTOMER_ID", "TX_TIME_SECONDS"])
if input_tensors:
        #---------- Test and Train split ------------
        input_x, y_tensor = input_tensors
        # clg.info(str(x_tensor.shape)+str(y_tensor.shape))
        # clg.info(str(x_tensor))
        # clg.info("pca comps >> " + str(int(0.1 * x_tensor.shape[2])))
        clg.info(">>"+str(x_tensor[:, :, :-1].shape))
        # x_components, c_variances = modules.DimRed(int(0.05 * x_tensor.shape[2])).pca(x_tensor[:, :, :-1].reshape(-1, x_tensor.shape[2]-1), 0.95)
        # input_x = np.hstack((x_components, x_tensor[:, :, -1].reshape(-1, 1))).reshape(-1, x_tensor.shape[1], x_components.shape[1]+1)
        # clg.info("components.shape:"+str(x_components.shape))
        # clg.info("input_x.shape:"+str(input_x.shape))
        # clg.info("c_variances:"+ str(len(c_variances)))
        # modules.Dumper(PROJ_DIRECTORY).dump([x_components, c_variances], 
                                                # "/data/tensors/pca/"+pre_proc.tensor_name, 
                                                # ["x_components", "c_variances"])
        
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
model.save(history,model.model_name)