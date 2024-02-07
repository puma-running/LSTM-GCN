import sys,os
sys.path.append(os.path.abspath('.'))
from util.Metrics import Metrics

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import torch
import torch.nn as nn
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib
matplotlib.use('TKAgg')
from util.Dictionary import Dictionary as Dic
from util.Ag_IndexsAccount import IndexsAccount as Account

torch.manual_seed(0)
np.random.seed(0)

input_window = 80
output_window = 1
# class LinearRegressionModel():
#     def __init__(self):
#         # Create linear regression object
#         self.regr = linear_model.LinearRegression()

class ModelRun():
    def __init__(self) -> None:
        self.model = linear_model.LinearRegression()
        self.metrics = Metrics()

    def create_inout_sequences(self, input_data, tw):
        inout_seq = []
        L = len(input_data)
        for i in range(L - tw):
            train_seq = input_data[i:i + tw]
            train_label = input_data[i + output_window:i + tw + output_window]
            inout_seq.append((train_seq, train_label))
        return torch.FloatTensor(inout_seq)
        
    def create_inout_sequences_time(self,input_data, tw):
        inout_seq = []
        L = len(input_data)
        for i in range(L - tw):
            train_seq = input_data[i:i + tw]
            train_label = input_data[i + output_window:i + tw + output_window]
            inout_seq.append((train_seq, train_label))
        return inout_seq

    def get_data(self, code):
        series = pd.read_csv(Dic.filename.format(code), usecols=['{}.close'.format(code)])
        series = series.values.reshape(-1)

        series_time = pd.read_csv(Dic.filename.format(code), usecols=['datetime'])
        series_time = series_time.values.reshape(-1)

        train_data = series[0:-1]
        time_data = series_time[0:-1]


        train_sequence = self.create_inout_sequences(train_data, input_window)
        train_sequence = train_sequence[:-output_window]

        time_sequence = self.create_inout_sequences_time(time_data, input_window)
        time_sequence = time_sequence[:-output_window]

        return train_sequence, np.array(time_sequence)  

    def get_batch(self, source):
        input = torch.stack([item[0] for item in source])
        target = torch.stack([item[1][-1] for item in source])
        return input, target
    def get_batch_times(self, source):
        input = np.stack([item[0][-1] for item in source])
        target = np.stack([item[1][-1] for item in source])
        return input, target

    def train(self, train_data):
        diabetes_X_train, diabetes_y_train = self.get_batch(train_data)
        # Train the model using the training sets
        self.model.fit(diabetes_X_train, diabetes_y_train)

    def evaluate(self):
        diabetes_X_test, diabetes_y_test = self.get_batch(val_data)
        # Make predictions using the testing set
        diabetes_y_pred = self.regr.predict(diabetes_X_test)

        # The coefficients
        # print("Coefficients: \n", self.regr.coef_)
        # The mean squared error
        # print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
        # The coefficient of determination: 1 is perfect prediction
        # print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))
    def tradeOnTestdatas(self, X_input, truth, test_result, times_input):
        '''
        在数据集上进行交易 # 交易回测
        '''
        # 交易回测
        # Dic.start_time = Dic.k_times[Dic.test_begin].split('.')[0]
        # Dic.end_time = Dic.k_times[Dic.test_end-1].split('.')[0]
        _trends = self.toTrends(X_input, truth, test_result)
        # 重新设置股票代码
        for i in range(0, len(_trends)):
            p_ask = _trends[i][0]
            p_bid = _trends[i][0]

            current_time =times_input[i]
            # print(current_time)
            price_close = X_input[i]
            # tupTime = time.localtime(_cur_times[i])#秒时间戳
            # stadardTime = time.strftime("%Y-%m-%d %H:%M:%S", tupTime)
            self.a1.trading(p_bid, p_ask, current_time, price_close)
        # 回测结束
        if len(_trends)!=0:
            current_time =times_input[i]
            price_close = X_input[i]
            self.a1.trading(88, 88, current_time, price_close)
            self.a1.printres()
    def toTrends(self, inputs, fTargets, outputs):
        '''
        返回趋势，返回判断预测的准确情况
        '''
        _inputs, _targets, _outputs = [], [], []
        trends =[]
        for x1, x2, x3 in zip(inputs, fTargets, outputs):
            _trends =[]
            '''规划到0-2'''
            npoint = x1*Dic.args.targetPoint
            """ 预测情况 """
            if x3-x1 >= npoint:
                _trends.append(2)
            elif x3-x1 <= -npoint:
                _trends.append(0)
            else:
                _trends.append(1)
            trends.append(_trends)
        return trends

    def plot_and_loss(self, data_source, times_source, epoch):
        diabetes_X_test, truth = self.get_batch(data_source)
        data_times, target_times = self.get_batch_times(times_source)
        # Make predictions using the testing set
        diabetes_y_pred = self.model.predict(diabetes_X_test)

        txt_test_result =""
        for index, s in enumerate(diabetes_y_pred):
            txt_test_result += "({},{:.2f})".format(index, s)
        txt_truth =""
        for index, s in enumerate(truth):
            txt_truth += "({},{:.2f})".format(index, s)
        X_input = np.stack([item[-1] for item in diabetes_X_test])
        test_result = diabetes_y_pred
        times_input = data_times
        X_input, test_result, truth = X_input.reshape(-1), test_result.reshape(-1), truth.reshape(-1)
        return -1, X_input, truth, test_result, times_input

        # return total_loss / i

# A50 stocks
# Dic.codes = sorted(Dic.codes,reverse=True)
# for code in Dic.codes:
#     # one stock
#     lr = MyLinearRegression()
#     lr.code = code
#     print("code={}".format(lr.code))
#     train_data, val_data = lr.get_data(code)
#     lr.train(train_data)
#     lr.evaluate(val_data)
#     lr.plot_and_loss()

# A50 stocks
Dic.codes = sorted(Dic.codes,reverse=True)
for code in Dic.codes:
    # 用于写交易日志
    Dic.code = code
    X_input_code, truth_code, test_result_code, times_input_code =[], [], [], []
    modelrun = ModelRun()
    modelrun.code = code
    print("code={}".format(modelrun.code))
    # torch.Size([800, 2, 80])
    # torch.Size([297, 2, 80])
    epochs = 1
    train_data_all, times_data_all = modelrun.get_data(code)
    n_len = train_data_all.shape[0]
    for i in range(22*10, n_len-22*2, 22*2):
        # 交易账户初始化
        modelrun.a1 = Account()
        modelrun.a1.balance.append(Dic.args.cashPerStock)
        Dic.log.info("当前月-{}".format(i))
        # Dic.log.info("{}".format(times_data_all[i][0][-1]))
        train_begin = i-22*10
        train_end = i
        test_begin = i
        # test_begin = i-input_window
        test_end = i+22*2
        train_data = train_data_all[train_begin:train_end,:,:]
        val_data = train_data_all[test_begin:test_end,:,:]
        times_data = times_data_all[test_begin:test_end,:,:]

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            modelrun.train(train_data)

            if epoch == epochs:
                val_loss, X_input, truth, test_result, times_input = modelrun.plot_and_loss(val_data, times_data, epoch)
                X_input_code.extend(X_input)
                truth_code.extend(truth)
                test_result_code.extend(test_result)
                times_input_code.extend(times_input)
            else:
                val_loss = modelrun.evaluate(val_data)
    # 交易
    # tradeOnTestdatas(self, X_input, truth, test_result, times_input):
    modelrun.tradeOnTestdatas(X_input_code, truth_code, test_result_code, times_input_code)
    modelrun.metrics.performance_metrics("Linear{}".format(modelrun.code), X_input_code, test_result_code, truth_code)
    modelrun.metrics.performance_values("Linear{}".format(modelrun.code), X_input_code, test_result_code, truth_code)
    plt.plot(test_result_code, color="red")
    plt.plot(truth_code, color="blue")
    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    plt.savefig('graph/Linear-epoch%d.png' % epoch)
    plt.close()