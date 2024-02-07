import torch
import torch.nn as nn
import numpy as np

arr1 = []
for i in range(10):
    arr1.append(i)
arr2 = []
for i in range(5):
    arr2.append(arr1)
arr2 = np.array(arr2)
arr3 = []
for i in range(6):
    arr3.append(arr2)

arrTensor = torch.FloatTensor(arr3)
arr3 = np.array(arr3)
arrTensor = torch.FloatTensor(arr3)
print(arr3)