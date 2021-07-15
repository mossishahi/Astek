import numpy as np
import tqdm

class Calculator:
    def __init__(self, data, 
    treshold, 
    sort_by = 'CUSTOMER_ID', 
    roll_base = 'CUSTOMER_ID'):
        self.data = data
        self.treshold = treshold
        self.roll_base = roll_base
        self.sort_by = sort_by

    def get_portion_size(self, n_portions = 100):
        counts = []
        for i in range(1, n_portions):
            counts.append((np.array(list(self.data.iloc[:int(i * (1/n_portions) * self.data.shape[0])].sort_values(by = self.sort_by)[self.roll_base].value_counts().to_dict().keys())) > self.treshold).sum())
        return counts