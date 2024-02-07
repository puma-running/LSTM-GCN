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

torch.manual_seed(0)
np.random.seed(0)

class Para:
    # CNN parameters
    # input_size = 8
    depth = 1
    attention_heads = 4
    embedding_dimension = 128
    drop_rate = 0.3 
    kernel_size = 16
    stride = 8
    # input_linear = embedding_dimension*9
    input_linear = embedding_dimension

    input_window = 80
    output_window = 1
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    epochs = 10
    lr = 0.001
    scaler = MinMaxScaler(feature_range=(-1, 1))

# 通常我们使用torch.arange(0, max_len)创建一个1维的列表，
# 然后通过unsqueeze(1),将列表变成形状为(max_len, 1)的数据，
# 然后再使用unsqueeze(0)，将列表形状变为(1, max_len, 1)的数据。
# 然后再将张量的第二维下标为奇数的部分，进行math.sin()函数的变换，
# 将张量第二维下标为偶数的部分进行math.cos()函数的变换。
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransAm(nn.Module):
    def __init__(self, feature_size=Para.embedding_dimension, num_layers=Para.depth):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.feature_size = feature_size
        self.conv1 = nn.Conv2d(1, 1, (1,Para.kernel_size), stride=Para.stride)
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=Para.attention_heads, dropout=Para.drop_rate)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(Para.input_linear, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # cnn
        # batch放在第0维
        src = src.permute(1,0,2)
        # 扩充维度
        src = torch.unsqueeze(src,1)
        # 转置
        src = src.permute(0,1,3,2)
        src= self.conv1(src)
        # for transformer
        src = torch.squeeze(src,1)
        src = src.permute(2,0,1)
        # transformer
        # if self.src_mask is None or self.src_mask.size(0) != len(src):
        #     device = src.device
        #     mask = self._generate_square_subsequent_mask(len(src)).to(device)
        #     self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        # output = output.permute(1,0,2).reshape(output.shape[1],-1)
        output = torch.squeeze(output.permute(1,0,2)[:,-1:],1)
        output = self.decoder(output)
        return output
        
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class ModelRun():   
    def __init__(self): 
        self.model = TransAm().to(Para.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=Para.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.95)
    
    def _create_inout_sequences(self, input_data, tw):
        inout_seq = []
        L = len(input_data)
        for i in range(L - tw):
            train_seq = input_data[i:i + tw]
            train_label = input_data[i + Para.output_window:i + tw + Para.output_window]
            inout_seq.append((train_seq, train_label))
        return torch.FloatTensor(inout_seq)

    def get_data(self, code):
        # SSE.600031
        series = pd.read_csv(Dic.filename.format(code), usecols=['{}.close'.format(code)])
        series = Para.scaler.fit_transform(series.values.reshape(-1, 1)).reshape(-1)

        train_samples = int(0.7 * len(series))
        train_data = series[:train_samples]
        test_data = series[train_samples:]


        train_sequence = self._create_inout_sequences(train_data, Para.input_window)
        train_sequence = train_sequence[:-Para.output_window]

        test_data = self._create_inout_sequences(test_data, Para.input_window)
        test_data = test_data[:-Para.output_window]

        return train_sequence.to(Para.device), test_data.to(Para.device)

    def get_batch(self, source, i, batch_size):
        seq_len = min(batch_size, len(source) - 1 - i)
        data = source[i:i + seq_len]
        input = torch.stack(torch.stack([item[0] for item in data]).chunk(Para.input_window, 1))  
        target = torch.stack(torch.stack([item[1] for item in data]).chunk(Para.input_window, 1))
        # 只取最后一个
        # target = torch.stack([item[-1] for item in target])
        return input, target

    def train(self, train_data):
        self.model.train()
        for batch_index, i in enumerate(range(0, len(train_data) - 1, Para.batch_size)):
            start_time = time.time()
            total_loss = 0
            data, targets = self.get_batch(train_data, i, Para.batch_size)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, targets[-1:][0])
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
            self.optimizer.step()

            total_loss += loss.item()
            log_interval = int(len(train_data) / Para.batch_size / 5)
            if batch_index % log_interval == 0 and batch_index > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                # print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.6f} | {:5.2f} ms | loss {:5.5f}'
                #     .format(epoch, batch_index, len(train_data) // Para.batch_size, self.scheduler.get_lr()[0], elapsed * 1000 / log_interval, cur_loss))

    def evaluate(self, data_source):
        self.model.eval() 
        total_loss = 0
        eval_batch_size = 1000
        with torch.no_grad():
            for i in range(0, len(data_source) - 1, eval_batch_size):
                data, targets = self.get_batch(data_source, i, eval_batch_size)
                output = self.model(data)
                total_loss += len(data[0]) * self.criterion(output, targets[-1:][0]).cpu().item()
        return total_loss / len(data_source)

    def plot_and_loss(self, data_source, epoch):
        self.model.eval()
        total_loss = 0.
        X_input = torch.Tensor(0)
        test_result = torch.Tensor(0)
        truth = torch.Tensor(0)
        with torch.no_grad():
            for i in range(0, len(data_source) - 1):
                data, target = self.get_batch(data_source, i, 1)
                output = self.model(data)
                total_loss += self.criterion(output, target[-1:][0]).item()
                test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0)
                X_input = torch.cat((X_input, data[-1].view(-1).cpu()), 0)
                truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)

        test_result= Para.scaler.inverse_transform(torch.unsqueeze(test_result,1))
        X_input= Para.scaler.inverse_transform(torch.unsqueeze(X_input,1))
        truth= Para.scaler.inverse_transform(torch.unsqueeze(truth,1))
        txt_test_result =""
        for index, s in enumerate(test_result):
            txt_test_result += "({},{:.2f})".format(index, s[0])
        txt_truth =""
        for index, s in enumerate(truth):
            txt_truth += "({},{:.2f})".format(index, s[0])

        metrics = Metrics()
        metrics.performance_metrics("CTTS{}".format(self.code), X_input.reshape(-1), test_result.reshape(-1), truth.reshape(-1))
        metrics.performance_values("CTTS{}".format(self.code), X_input.reshape(-1), test_result.reshape(-1), truth.reshape(-1))

        plt.plot(test_result, color="red")
        plt.plot(truth, color="blue")
        plt.grid(True, which='both')
        plt.axhline(y=0, color='k')
        plt.savefig('graph/transformer-epoch%d.png' % epoch)
        plt.close()
        return total_loss / i

# A50 stocks
Dic.codes = sorted(Dic.codes,reverse=True)
for code in Dic.codes:
    runmodel = ModelRun()
    runmodel.code = code
    print("code={}".format(runmodel.code))
    train_data, val_data = runmodel.get_data(runmodel.code)

    # one stock
    for epoch in range(1, Para.epochs + 1):
        epoch_start_time = time.time()
        runmodel.train(train_data)

        if (epoch % 10 is 0):
            if epoch == Para.epochs:
                val_loss = runmodel.plot_and_loss(val_data, epoch)
            else:
                val_loss = runmodel.plot_and_loss(val_data, epoch)
        else:
            val_loss = runmodel.evaluate(val_data)
        runmodel.scheduler.step()