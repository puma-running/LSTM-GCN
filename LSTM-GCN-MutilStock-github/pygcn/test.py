import torch
torch.manual_seed(2)
A = torch.randn(4,2)
print(A)
B = A.uniform_(0.5,1)
print(B)
#结果
# tensor([[ 0.6849, -2.6560],
#         [ 0.3943,  0.1994]])
import numpy as np
import pandas as pd
m=np.random.random((6,6))
df=pd.DataFrame(m)
print(df.corr())

import torch
import torch.nn as nn
crit=nn.MSELoss()#均方损失函数
target = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
pred= torch.FloatTensor([[7, 8, 9], [8, 4, 3]])
cost=crit(pred,target)#将pred,target逐个元素求差,然后求平方,再求和,再求均值,
print(cost)