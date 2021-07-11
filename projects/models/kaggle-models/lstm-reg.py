import numpy as np
import pandas as pd
import pickle
import os, glob
import logging
import seaborn as sns
from keras import optimizers
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.recurrent import LSTM
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
import matplotlib.pyplot as plt
import logging
from keras.callbacks import LambdaCallback


logging.basicConfig(filename='/home/m.shah/app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('train_logger')
logging.warning('This will get logged to a file')




# ## Loading Data

# In[7]:


dfs = []

for filename in glob.glob(os.path.join(path, "../../data/simulated-data-raw/", "data", "*.pkl")):
    with open(filename, 'rb') as f:
        temp = pd.read_pickle(f)
        dfs.append(temp)
df = pd.DataFrame()
df = df.append(dfs)


# In[8]:


#df.head()


# In[9]:


#df.columns


# In[10]:


df = df.iloc[:, [0, 1, 2, 3, 4, 5, 6, 8, 7]]


# In[11]:


#df.head()


# ## Preprocessing

# In[12]

print("file Read Done!")
class Preprocessor:    
    def __init__(self, input_data):
        self.data = input_data
    
    def pre_process(self, 
                    feature_columns,
                    label_columns,
                    window_size = 128,
                    numericals = None,
                    categoricals = None,
                    test_train_split = 0.7,
                    imbalanced = False):
        # ---------- 
        self.window_size = window_size
        self.test_train_split = test_train_split
        self.data_x = pd.DataFrame(self.data[feature_columns], dtype = 'float32')
        self.data_y = pd.DataFrame(self.data[label_columns], dtype = 'float32')
        self.n_features = len(feature_columns)
        self.n_labels = len(label_columns)
        # ---------- Categorical features, OneHotEncoding
        print(self.data_x.shape)
        if categoricals is not None:
            self.encode(categoricals)
            with open("./log.txt", 'a') as log:
                log.write("- - - - Categorical features Encoded - - - -\n")
        print(self.data_x.shape)
        # ---------- Standardization
        self.scaler(numericals)
        # ---------- Rolling X and Y Tensors
        print("======== Making X and Y Tensors ==========")
        self.x_tensor, self.y_tensor = self.roll(self.window_size)
        print("Input X by shape:", self.data_x.shape,"is rolled to X Tensor: ", self.x_tensor.shape)
        with open("./log.txt", 'a') as log:
            log.write("X and Y tensors are Rolled \n")

        self.train_idx = np.random.choice(self.x_tensor.shape[0], 
                                      int(self.x_tensor.shape[0]*self.test_train_split), replace=False)
                      
        self.X_train = self.x_tensor[self.train_idx, :, :]
        self.X_test = np.delete(self.x_tensor, self.train_idx, axis=0)
        self.y_train = self.y_tensor[self.train_idx, ]
        self.y_test = np.delete(self.y_tensor, self.train_idx, axis=0)
        with open("./log.txt", 'a') as log:
            log.write("Preprocessing Done! \n")

        return self.X_train, self.X_test, self.y_train, self.y_test
                
    def encode(self, categoricals):
        if categoricals[0] is not None:
            self.data_x = pd.get_dummies(self.data_x, columns = categoricals[0])
            self.n_features = self.data_x.shape[1]
        if categoricals[1] is not None:
            self.data_y = pd.get_dummies(self.data_y, columns = categoricals[1])
    
    def scaler(self, numericals):
        scaler = MinMaxScaler()
        self.data_x[numericals[0]] = scaler.fit_transform(self.data_x[numericals[0]])
        self.data_y[numericals[1]] = scaler.fit_transform(self.data_x[numericals[1]])

    def roll(self, window_size):
        starts = np.array(range(self.data_x.shape[0]-(window_size)))
        if os.path.isfile('ix_tensor.pickle'):
            print("- - - -  X Tensor founded on Local Drive - - - - ")
            print("- - - -  Reading X TENSOR - - - - ")
            with open('ix_tensor.pickle', 'rb') as file:
                ix_tensor = joblib.load(file)
            with open('iy_tensor.pickle', 'rb') as file:
                iy_tensor = joblib.load(file)
        else:
            ix_tensor = np.zeros([(self.data_x.shape[0]-(window_size))*window_size, 
                                  self.n_features], dtype = 'float32')
            iy_tensor = np.zeros((0, self.n_labels), dtype = 'float32')
            
            #Rolling Loop for making ix_tensor
            for i in tqdm(range(self.data_x.shape[0]-(window_size))):
                s = np.array(self.data_x[i:i+window_size], dtype='float32')
                ix_tensor[(window_size*i):(window_size*(i+1)), :] = s
                iy_tensor = np.vstack((iy_tensor, self.data_y.iloc[i+window_size, ]))
            ix_tensor = ix_tensor.reshape(-1, window_size, np.shape(ix_tensor)[1])
            print("- - - - Writing X Tensor on Drive- - - -")
            with open('ix_tensor.pickle', 'wb') as file:
                joblib.dump(ix_tensor, file)
            with open('iy_tensor.pickle', 'wb') as file:
                joblib.dump(iy_tensor, file) 
        print("type::",type(iy_tensor[0][0]))
        return ix_tensor, iy_tensor


# In[14]:


#Feature Define
if "WEEK_DAY" not in df.columns:
    df.insert(7, "WEEK_DAY", df["TX_DATETIME"].apply(lambda x : x.weekday()))

#Feature Selection
selected_features = ["CUSTOMER_ID",
                     "TERMINAL_ID",
                     'TX_AMOUNT', 
                     'TX_TIME_SECONDS', 
                     'TX_TIME_DAYS', 
                     "WEEK_DAY", 
                     "TX_FRAUD_SCENARIO"]


# In[16]:


pr = Preprocessor(df.iloc[:4000, :10])
train_x, test_x, train_y, test_y = pr.pre_process(selected_features, 
               ["TX_AMOUNT"],
               numericals = [["TX_AMOUNT", "TX_TIME_SECONDS",'TX_TIME_DAYS'],["TX_AMOUNT"]],
              categoricals = [["CUSTOMER_ID", "TERMINAL_ID", "WEEK_DAY", "TX_FRAUD_SCENARIO"],None])
               


class LSTM_REGRESSION:
    def __init__(self,
                 input_shape,
                 lstm_units = 50,
                 n_outputs = 1,
                 optimizer = 'adam',
                loss = 'mse',
                metrics = 'mae'):
        #Model Layers
        self.model = Sequential()
        self.model.add(LSTM(lstm_units, input_shape=input_shape, return_sequences = True))#, return_sequences=True))            
        self.train_log = open('train.log', mode='wt', buffering=1)
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(lstm_units, return_sequences = True))            
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(lstm_units))            
        self.model.add(Dropout(0.2))
        self.model.add(Dense(n_outputs))

        self.model.compile(loss=loss, 
                           optimizer=optimizer, 
                           metrics=metrics) #optimizer='rmsprop'
        self.model.build()
        print("Model Summary: \n\n", self.model.summary())
        self.model
        
    def train(self, train_x, train_y, batch_size = 256, epochs  =2):
        train_logger = LambdaCallback(on_epoch_end = lambda epoch, logs: self.train_log.write(str({'epoch': epoch, 'loss': logs['loss'], 'val_loss':logs['val_loss'], 'mae':logs['mae']})), on_train_end = lambda logs: self.train_log.close())
        result = self.model.fit(train_x, train_y,batch_size = batch_size,validation_split = 0.2,epochs = epochs,callbacks = [train_logger])
        return result
    def predict(self, test_x):
        return self.model.predict(test_x)


# In[171]:
normal_model = LSTM_REGRESSION(train_x.shape[1:], 
                               lstm_units = 80, 
                               n_outputs = train_y.shape[1], 
                               optimizer = optimizers.Adam(learning_rate = 0.01))

n_history = normal_model.train(train_x, train_y, epochs = 100)

with open("n_history_n.pickle","wb") as file:
	pickle.dump(n_history.history,file)


