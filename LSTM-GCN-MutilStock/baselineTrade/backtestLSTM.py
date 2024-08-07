import sys,os
sys.path.append(os.path.abspath('.'))
from util.Metrics import Metrics

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
batch_size = 64
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.model_type = 'LSTM'
        self.lstm = nn.LSTM(1, 80, 1, batch_first=True)
        self.linear = nn.Linear(80, 1)

    def init_weights(self):
        initrange = 0.1
        self.lstm.bias.data.zero_()
        self.lstm.permute_hidden

    def forward(self, src):
        output, (hn, cn) = self.lstm(src)
        result = self.linear(hn[0,:,:])
        return result
        
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class ModelRun():
    def __init__(self) -> None:
        self.model = LSTMModel().to(device)
        self.criterion = nn.MSELoss()
        lr = 0.001
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.95)
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.metrics = Metrics()

    def create_inout_sequences(self,input_data, tw):
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
        series = self.scaler.fit_transform(series.values.reshape(-1, 1)).reshape(-1)

        series_time = pd.read_csv(Dic.filename.format(code), usecols=['datetime'])
        series_time = series_time.values.reshape(-1)

        train_data = series[0:-1]
        time_data = series_time[0:-1]
        train_sequence = self.create_inout_sequences(train_data, input_window)
        train_sequence = train_sequence[:-output_window]

        time_sequence = self.create_inout_sequences_time(time_data, input_window)
        time_sequence = time_sequence[:-output_window]

        return train_sequence.to(device), np.array(time_sequence)

    def get_batch(self, source, i, batch_size):
        seq_len = min(batch_size, len(source) - 1 - i)
        data = source[i:i + seq_len]
        input = torch.unsqueeze(torch.stack([item[0] for item in data]),2)
        target = torch.unsqueeze(torch.stack([item[1] for item in data]),2)
        return input, target

    def get_batch_times(self, source, i, batch_size):
        seq_len = min(batch_size, len(source) - 1 - i)
        data = source[i:i + seq_len]
        input = np.expand_dims(np.stack([item[0] for item in data]),2)
        target = np.expand_dims(np.stack([item[1] for item in data]),2)
        return input, target

    def train(self, train_data):
        self.model.train()
        for batch_index, i in enumerate(range(0, len(train_data) - 1, batch_size)):
            start_time = time.time()
            total_loss = 0
            data, targets = self.get_batch(train_data, i, batch_size)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, targets[:,-1])
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            log_interval = int(len(train_data) / batch_size / 5)
            if batch_index % log_interval == 0 and batch_index > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time

    def evaluate(self, data_source):
        self.model.eval() 
        total_loss = 0
        eval_batch_size = 1000
        with torch.no_grad():
            for i in range(0, len(data_source) - 1, eval_batch_size):
                data, targets = self.get_batch(data_source, i, eval_batch_size)
                output = self.model(data)
                total_loss += len(data[0]) * self.criterion(output, targets[:,-1]).cpu().item()
        return total_loss / len(data_source)

    def plot_and_loss(self, data_source, times_source, epoch):
        self.model.eval()
        total_loss = 0.
        X_input = torch.Tensor(0)
        test_result = torch.Tensor(0)
        truth = torch.Tensor(0)
        times_input = []
        with torch.no_grad():
            for i in range(0, len(data_source) - 1):
                data, target = self.get_batch(data_source, i, 1)
                data_times, target_times = self.get_batch_times(times_source, i, 1)
                output = self.model(data)
                total_loss += self.criterion(output, target[:,-1]).item()
                test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0)
                X_input = torch.cat((X_input, data[:,-1].view(-1).cpu()), 0)
                truth = torch.cat((truth, target[:,-1].view(-1).cpu()), 0)
                times_input.append(data_times[0,-1,0][0:19])

        X_input= self.scaler.inverse_transform(torch.unsqueeze(X_input,1))
        test_result= self.scaler.inverse_transform(torch.unsqueeze(test_result,1))
        truth= self.scaler.inverse_transform(torch.unsqueeze(truth,1))
        txt_test_result =""
        for index, s in enumerate(test_result):
            txt_test_result += "({},{:.2f})".format(index, s[0])
        txt_truth =""
        for index, s in enumerate(truth):
            txt_truth += "({},{:.2f})".format(index, s[0])

        X_input, test_result, truth = X_input.reshape(-1), test_result.reshape(-1), truth.reshape(-1)
        return total_loss / i, X_input, truth, test_result, times_input

    def tradeOnTestdatas(self, X_input, truth, test_result, times_input):
        _trends = self.toTrends(X_input, truth, test_result)
        for i in range(0, len(_trends)):
            p_ask = _trends[i][0]
            p_bid = _trends[i][0]

            current_time =times_input[i]
            price_close = X_input[i]
            self.a1.trading(p_bid, p_ask, current_time, price_close)
        if len(_trends)!=0:
            current_time =times_input[i]
            price_close = X_input[i]
            self.a1.trading(88, 88, current_time, price_close)
            self.a1.printres()

    def toTrends(self, inputs, fTargets, outputs):
        '''
        Returns the trend, returns the accuracy of the prediction
        '''
        _inputs, _targets, _outputs = [], [], []
        trends =[]
        for x1, x2, x3 in zip(inputs, fTargets, outputs):
            _trends =[]
            npoint = x1*Dic.args.targetPoint
            if x3-x1 >= npoint:
                _trends.append(2)
            elif x3-x1 <= -npoint:
                _trends.append(0)
            else:
                _trends.append(1)
            trends.append(_trends)
        return trends
# A50 stocks
Dic.codes = sorted(Dic.codes,reverse=True)
for code in Dic.codes:
    Dic.code = code
    X_input_code, truth_code, test_result_code, times_input_code =[], [], [], []
    modelrun = ModelRun()
    modelrun.code = code
    print("code={}".format(modelrun.code))
    epochs = 1
    train_data_all, times_data_all = modelrun.get_data(code)
    n_len = train_data_all.shape[0]
    for i in range(22*10, n_len-22*2, 22*2):
        modelrun.a1 = Account()
        modelrun.a1.balance.append(Dic.args.cashPerStock)
        Dic.log.info("Current month-{}".format(i))
        train_begin = i-22*10
        train_end = i
        test_begin = i
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
            modelrun.scheduler.step()

    # Trading
    modelrun.tradeOnTestdatas(X_input_code, truth_code, test_result_code, times_input_code)
    modelrun.metrics.performance_metrics("LSTM{}".format(modelrun.code), X_input_code, test_result_code, truth_code)
    modelrun.metrics.performance_values("LSTM{}".format(modelrun.code), X_input_code, test_result_code, truth_code)
    plt.plot(test_result_code, color="red")
    plt.plot(truth_code, color="blue")
    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    plt.savefig('graph/LSTM-epoch%d.png' % epoch)
    plt.close()