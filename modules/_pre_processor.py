import joblib
import os, glob
import numpy as np
import pandas as pd
import pickle
import logging

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import joblib
from keras.callbacks import CSVLogger
from ._vc import VC
from ._dumper import Dumper
from ._dim_red import DimRed

class Preprocessor:    
    def __init__(self, 
        input_data, 
        portion,
        loggers, 
        path = os.path.abspath("./data/tensors/") + "/"):
        self.data = input_data.iloc[:int(portion*input_data.shape[0])]
        vc = VC()
        self.version = vc.write_version()
        self.path = path 
        self.clg, self.flg = loggers
        self.clg.info("Code Version: "+ str(self.version))
        self.clg.info("Preprocessor: data with Shape:" + \
            str(input_data.shape)+ " trimmed to " +str(self.data.shape) +"\n")

    def pre_process(self, 
                    feature_columns,
                    label_columns,
                    window_size = 128,
                    numericals = None,
                    categoricals = None,
                    test_train_split = 0.7,
                    roll_base = 'time',
                    drop_rollbase = False,
                    imbalanced = False,
                    dimension_reduction = None):
        # ---------- 
        self.features = feature_columns
        self.labels = label_columns
        self.window_size = window_size
        self.drop_rollbase = drop_rollbase
        self.roll_base = roll_base
        self.customers = np.array([])
        self.test_train_split = test_train_split
        if 'TX_DATETIME' in feature_columns:
            self.data.loc[:]['TX_DATETIME'] = self.data.loc[:]['TX_DATETIME'].values.astype(float) 
        # ---------- Standardization ----------
        self.scaler(numericals)
        
        # ---------- Categorical features >> OneHotEncoding -------------
        if categoricals:
            self.clg.info("Raw data shape:" + str(self.data.shape))
            self.encode(categoricals)
            self.clg.info("One-hotted data shape:" + str(self.data.shape))

        #----------- Dimension Reduction -----------
        if dimension_reduction:
            high_dimensions = categoricals
            self.clg.info(self.data.columns)
            self.clg.info("---- Dimension Reduction Started ----")
            self.dimension_reduction(high_dimensions, params = [700], method = dimension_reduction)
            self.features.append("V") #columns of latent space got from AE are called V#
            self.clg.info(self.data.columns) 
            Dumper(self.path).dump([self.data], str(self.version) + "-dim_red-", ["input_data_Ldim"])
            self.clg.info("---- Dimension Reduction Done----")

        # ---------- Rolling X and Y Tensors ---------
        tensors, message = self.roll(self.window_size)
        if tensors:
            self.x_tensor, self.y_tensor = tensors
            self.clg.info(message)
            input_tensors = self.x_tensor, self.y_tensor
        else:
            return None, message
        # #---------- Test and Train split ------------
        # self.train_idx = np.random.choice(self.x_tensor.shape[0], 
        #                               int(self.x_tensor.shape[0]*self.test_train_split), replace=False)                      
        
        # self.X_train = self.x_tensor[self.train_idx, :, :]
        # self.X_test = np.delete(self.x_tensor, self.train_idx, axis=0)
        # self.y_train = self.y_tensor[self.train_idx, ]
        # self.y_test = np.delete(self.y_tensor, self.train_idx, axis=0)
        # message =  "-------------------- Preprocessing " + "Version-" + str(self.version) + " Done! ---------------------- \n"
        # input_tensors = [self.X_train, self.X_test, self.y_train, self.y_test]
        # Dumper(self.path + "train_tensors/").dump(input_tensors, 
        #                        self.tensor_name,
        #                        "self.X_train, self.X_test, self.y_train, self.y_test".replace(",","").replace("self.", "").split(" "))
        return input_tensors, message
                
    def encode(self, categoricals):
        if categoricals is not None:
            self.data = pd.get_dummies(self.data, columns = categoricals)
    
    def scaler(self, numericals):
        scaler = MinMaxScaler()
        self.data.loc[:, numericals] = scaler.fit_transform(self.data.loc[:, numericals])
        # self.clg.info(str(self.data.loc[:][numericals].iloc[[0, 100, 500]][:]) + "\n")
        # self.clg.info("shape-test" + str(self.data.shape))
        self.flg.info(str(numericals)+ " >> have been Normalized.")

    def roll(self, window_size):
        
        valid_customers = (np.array(list(self.data[self.roll_base[0]].value_counts().to_dict().values())) > window_size).sum()
        if valid_customers:
            self.clg.info("No. Valid Customers:" + str(((np.array(list(self.data[self.roll_base[0]].value_counts().to_dict().values())) > window_size).sum())))
        else:
            message = " There is no Customer having at least " + str(window_size) + " Transaction \n"
            return None, message
        x_filter = [col for col in self.data.columns if col.startswith(tuple(self.features))]
        y_filter = [col for col in self.data.columns if col.startswith(tuple(self.labels))]
        ix_tensor = np.zeros([(self.data.shape[0] - (window_size)) * window_size, len(x_filter)], dtype = 'float32')
        iy_tensor = np.zeros((0, len(y_filter)), dtype = 'float32')
        tensor_name = "W"+str(window_size)+"X"+str(len(x_filter))+"Y"+str(len(y_filter))+"_"
        # print(">> . . .", self.data[self.roll_base[0]].value_counts().to_dict())
        if self.roll_base == 'time':
            tensor_name = "T-" + tensor_name
            tensor_name = "V-" + str(self.version) + tensor_name
            temp_filter = x_filter.pop(x_filter.index('TX_AMOUNT'))
            x_filter = x_filter + [temp_filter]
            self.dg_x = dg.loc[:][x_filter]
            self.dg_y = self.data.loc[:][y_filter]
            # - - - - - Check Drive files - - - - 
            if os.path.isfile(self.path + tensor_name +'X_tensor.pickle'):
                with open(self.path + str(tensor_name + +'X_tensor.pickle'), 'rb') as file:
                    ix_tensor = joblib.load(file)
                with open(self.path + str(tensor_name + +'Y_tensor.pickle'), 'rb') as file:
                    iy_tensor = joblib.load(file)
                message = "- - -> X-Tensor with shape: " + str(ix_tensor.shape) + " and" + " Y-Tensor with shape" + str(iy_tensor.shape) + "found on Local Drive <- - -"
            else:
                # Rolling Loop for making ix_tensor
                for i in tqdm(range(self.dg_x.shape[0]-(window_size))):
                    s = np.array(self.dg_x[i:i+window_size], dtype='float32')
                    ix_tensor[(window_size*i):(window_size*(i+1)), :] = s
                    iy_tensor = np.vstack((iy_tensor, self.dg_y.iloc[i+window_size, ]))
                ix_tensor = ix_tensor.reshape(-1, window_size, np.shape(ix_tensor)[1])
                self.clg.info( "- - - - Writing X Tensor on Drive- - - -")
                with open(self.path + tensor_name + +'X_tensor.pickle', 'wb') as file:
                    joblib.dump(ix_tensor, file)
                with open(self.path + tensor_name + +'Y_tensor.pickle', 'wb') as file:
                    joblib.dump(iy_tensor, file) 
                message = "Input X by shape:" + str(self.data.shape) + "rolled to X Tensor by Shape:" + str(ix_tensor.shape)
        else:
            tensor_name = "R-"+tensor_name
            tensor_name = "V" + str(self.version) + "-" + tensor_name
            if os.path.isfile(self.path + tensor_name + 'X_tensor.pickle'):
                with open((self.path + str(tensor_name) + 'X_tensor.pickle'), 'rb') as file:
                    ix_tensor = joblib.load(file)
                with open((self.path + str(tensor_name) + 'Y_tensor.pickle'), 'rb') as file:
                    iy_tensor = joblib.load(file)
                message = "- - -> X-Tensor with shape: " + str(ix_tensor.shape) + " and" + " Y-Tensor with shape" + str(iy_tensor.shape) + "found on Local Drive <- - -"
            else:
                dg = self.data.sort_values(by = self.roll_base)
                temp_filter = x_filter.pop(x_filter.index('TX_AMOUNT')) #put amount column at the end of df
                x_filter = x_filter + [temp_filter]
                self.dg_x = dg.loc[:][x_filter]                
                self.dg_y = dg.loc[:][y_filter]

                self.flg.info("normal-check-X:" + str((self.dg_x > 1.01).sum()) + "\n") #check whether numericals are normalaized
                self.flg.info("normal-check-Y:" + str((self.dg_y > 1.01).sum()) + "\n")
                self.flg.info("dgCols" + str(self.dg_x.columns))
                with open("dg.pickle", "w+b") as f:
                    pickle.dump(dg, f)
                with open("dg_x.pickle", "w+b") as f:
                    pickle.dump(self.dg_x, f)
                with open("dg_y.pickle", "w+b") as f:
                    pickle.dump(self.dg_y, f)

                del(dg)

                # ------- Rolling Looop -----
                i = 0
                j = 0
                while (i < self.dg_x.shape[0]-(window_size)):
                    if not i %1000: # progress bar
                        self.clg.info(str(i)+" /" + str(self.dg_x.shape[0]-(window_size)))
                    if self.dg_x[self.roll_base[0]].iloc[i:i + window_size].eq(self.dg_x[self.roll_base[0]].iloc[i + window_size]).all():
                        s = np.array(self.dg_x.iloc[i:i + window_size], dtype='float32')
                        ix_tensor[(window_size*j):(window_size*(j+1)), :] = s
                        iy_tensor = np.vstack((iy_tensor, self.dg_y.iloc[i+window_size,:]))
                        j += 1
                        i += 1
                    else:
                        i += 1
                        continue
                # -------- Drop RoleBase Column -----
                if self.drop_rollbase:
                    ix_tensor = np.delete(ix_tensor, self.dg_x.columns.get_loc(self.roll_base[0]), axis = 1)
                ix_tensor = ix_tensor[:len(iy_tensor)*window_size ,:].reshape(-1, window_size, ix_tensor.shape[1])
                iy_tensor = iy_tensor.reshape(-1, 1)
                # ------- Write Tensors on Drive ------
                with open(self.path + tensor_name +'X_tensor.pickle', 'wb') as file:
                    joblib.dump(ix_tensor, file)
                with open(self.path + tensor_name +'Y_tensor.pickle', 'wb') as file:
                    joblib.dump(iy_tensor, file) 
                message = "Input X by shape:" + str(self.data.shape) + "rolled to X Tensor by Shape:" + str(ix_tensor.shape)
                self.tensor_name = tensor_name
        return [ix_tensor, iy_tensor], message


    def dimension_reduction(self, high_dimensions, params = [0.9], method = "pca"):
        h_cols = [col for col in self.data.columns if col.startswith(tuple(high_dimensions))] 
        l_cols = [col for col in self.data.columns if not col.startswith(tuple(high_dimensions))] 
        dim_red = DimRed(self.data[h_cols], str(self.version) + "-dim_red-")
        dr_res, _, _ = getattr(dim_red, method)(*params)
        dr_columns = ["V"+str(i) for i in range(dr_res.shape[1])]
        dr_df = pd.DataFrame(dr_res, columns = dr_columns)
        self.data = pd.concat((dr_df, self.data[l_cols]), axis = 1)