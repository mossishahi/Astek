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
#tensorflow Logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
#project-directory
PROJ_DIRECTORY = os.getcwd()
#log, clg: console - flg:file
clg, flg = modules.MyLog().getLogger()
#----------------------------------------------------------------------------------


# Loading Data
sim_df = modules.DataLoader(base = PROJ_DIRECTORY).load_data("simulated-data-raw")

#Feature Define
if "WEEK_DAY" not in sim_df.columns:
    sim_df.insert(7, "WEEK_DAY", sim_df["TX_DATETIME"].apply(lambda x : x.weekday()))
#Feature Selection
selected_features = ["CUSTOMER_ID",
                     "TERMINAL_ID",
                     'TX_AMOUNT', 
                     'TX_TIME_SECONDS', 
                     'TX_TIME_DAYS', 
                     "WEEK_DAY", 
                     "TX_FRAUD_SCENARIO"]
                     
flg.info('selected Features are:' + str(selected_features))

# calc = modules.Calculator(sim_df, 8).get_portion_size()

#Preprocess Data
pre_proc = modules.Preprocessor(sim_df, 0.006, [clg, flg])
input_tensors, message = pre_proc.pre_process(selected_features, ['TX_AMOUNT'], 
                    numericals = [["TX_AMOUNT", "TX_TIME_SECONDS",'TX_TIME_DAYS'],["TX_AMOUNT"]],
                    categoricals = [["TERMINAL_ID", "WEEK_DAY", "TX_FRAUD_SCENARIO"],None], 
                    window_size = 8,
                    roll_base = ["CUSTOMER_ID", "TX_TIME_SECONDS"])
if input_tensors:
    train_x, test_x, train_y, test_y = input_tensors
    clg.info(str(train_x.shape) + str(test_x.shape) + str(train_y.shape) + str(test_y.shape))
else:
    clg.warning(message)
    flg.warning(message)
    raise Exception('input Data is empty')

#feed data to Model
model = models.REGRESSION(train_x.shape[1:], n_outputs=train_y.shape[1])
history = model.train(train_x, train_y)
model.save(history)