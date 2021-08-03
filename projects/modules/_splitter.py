class Splitter:
    def __init__(self):
        pass
    def split(slef, data, test_portion):
        #---------- Test and Train split ------------
        train_idx = np.random.choice(data.shape[0], 
                                      int(data.shape[0] * test_portion), replace=False)                      
        
        self.X_train = data[train_idx, :, :]
        self.X_test = np.delete(data, train_idx, axis=0)
        self.y_train = self.y_tensor[train_idx, ]
        self.y_test = np.delete(self.y_tensor, train_idx, axis=0)

