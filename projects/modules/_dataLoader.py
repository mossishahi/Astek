import pandas as pd
import os
import glob

class DataLoader:
    def __init__(self, base):
        self.base = base
        pass
    def load_data(self, name):
        dfs = []
        for filename in glob.glob(os.path.join(self.base, "data", name, "*.pkl")):
            # print(filename)
            with open(filename, 'rb') as f:
                temp = pd.read_pickle(f)
                dfs.append(temp)
        df = pd.DataFrame()
        df = df.append(dfs)
        df = df.iloc[:, [0, 1, 2, 3, 4, 5, 6, 8, 7]]
        return df
        