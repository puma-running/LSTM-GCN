import sys, os
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath('.'))
from util.Dictionary import Dictionary as Dic

Dic.initParameter("computerA50")
Dic.initLog()
Dic.klines = dict()
for code in Dic.codes:
    # 股票路径
    filename = Dic.filename.format(code)
    if os.path.exists(filename) == False:
        Dic.log.info("数据不存在{}".format(code))
        continue
    Dic.log.info("数据装载{}".format(code))
    df = pd.read_csv(filename)
    use_cols = ["datetime","{}.open".format(code),"{}.high".format(code),"{}.low".format(code),"{}.close".format(code)
        , "{}.volume".format(code),"{}.open_oi".format(code),"{}.close_oi".format(code)]
    klines = df[use_cols].values
    print(klines[:,0][0])
    print(klines[:,0][-1])
    Dic.klines[code] = klines[:,4]/klines[:,4][0]

res = []
for value in Dic.klines.values():
    if len(res)>0:
        res = list(map(lambda x, y: x + y, value, res))
    else:
        res = value
txt = ""
for i in range(len(res)):
    txt = txt + "({},{:.2f})".format(i+1, res[i])
print(0)