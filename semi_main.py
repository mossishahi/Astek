import os
print(os.getcwd())
import pandas as pd
import numpy as np

a = [[[1, 2, 4], [2, 3, 4]], [[3, 4, 4], [5, 6, 4]], [[7,8, 4], [9, 10, 4]]]

ar = np.array(a)
print(ar.shape)
print(ar.reshape(-1, ar.shape[2]).shape)
print(ar[0:0])

# b = [1, 2, 3, 4, 'a']
# [b.pop(4).append(['b'])]
# print(b)
b = [1]
for i in range(10):
    b = np.append(b, "c"+str(i))
print(b)

bb = ["c"+str(i) for i in range(len(ar[0]))]
print(bb)
df = pd.DataFrame()