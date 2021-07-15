from keras import optimizers
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.recurrent import LSTM

class SIMPLE_LSTM:
    def __init__(self,
                 input_shape,
                 loss,
                 metrics,
                 drop_out = None,
                 lstm_layers = [50, 50, 50],
                 n_outputs = 1,
                 optimizer = 'adam'):

        # Model Layers
        self.model = Sequential()
        self.model.add(LSTM(lstm_layers[0], input_shape = input_shape, return_sequences = (len(lstm_layers)>1)))
        for i in range(1, len(lstm_layers)):
            self.mode.add(LSTM(lstm_layers[i], return_sequences = False))
            if drop_out is not None:
                self.model(Dropout(drop_out))
        
        self.model.compile(loss=loss, 
                           optimizer=optimizer, 
                           metrics=metrics) #optimizer='rmsprop'
        self.model.build()
        print("Model Summary: \n\n", self.model.summary())
        
        #model's name
        self.model_name = "L" + "".join(map(str,lstm_layers)) + "-"
        self.model_name += "W" + str(input_shape[1])
        self.model_name += ">>" + str(n_outputs)
        
    # def train(self, train_x, train_y, batch_size = 100, epochs  =1):
    #     csv_logger = CSVLogger('/home/m.shah/projects/models/kaggle-models/training.log', append=True, separator=';')
    #     result = self.model.fit(train_x,
    #                             train_y, 
    #                             batch_size = batch_size,
    #                             validation_split = 0.2,
    #                             epochs = epochs,
    #                            callbacks=[csv_logger])
    #     return result
    # def predict(self, test_x):
    #     return self.model.predict(test_x)
    # def save(self, hitory):
    #     self.model.save(self.model_name)
    #     with open(str("h_"+self.model_name), 'wb') as file:
    #         joblib.dump(history.history, file)