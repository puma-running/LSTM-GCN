import sys,os
sys.path.append(os.path.abspath('.'))
from util.Metrics import Metrics

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib
matplotlib.use('TKAgg')
from util.Dictionary import Dictionary as Dic
np.random.seed(0)
depth = 1
attention_heads = 4
embedding_dimension = 128
drop_rate = 0.3 
kernel_size = 16
stride = 8
input_linear = embedding_dimension

input_window = 80
output_window = 1
batch_size = 64

class MyRandomForest():
    def __init__(self) -> None:
        # Create linear regression object
        self.regr = RandomForestRegressor(n_estimators=10)

    def create_inout_sequences(self, input_data, tw):
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

        train_samples = int(0.7 * len(series))
        train_data = series[:train_samples]
        test_data = series[train_samples:]

        train_sequence = self.create_inout_sequences(train_data, input_window)
        train_sequence = train_sequence[:-output_window]

        test_data = self.create_inout_sequences(test_data, input_window)
        test_data = test_data[:-output_window]

        return train_sequence, test_data  

    def get_batch(self, source):
        input = [item[0].tolist() for item in source]
        target = [item[1][-1].tolist() for item in source]
        return input, target

    def train(self, train_data):

        diabetes_X_train, diabetes_y_train = self.get_batch(train_data)
        diabetes_X_train, diabetes_y_train = np.array(diabetes_X_train), np.array(diabetes_y_train)
        # Train the model using the training sets
        self.regr.fit(diabetes_X_train, diabetes_y_train)

    def evaluate(self, data_source):
        diabetes_X_test, diabetes_y_test = self.get_batch(val_data)
        # Make predictions using the testing set
        diabetes_y_pred = self.regr.predict(diabetes_X_test)

    def plot_and_loss(self):
        diabetes_X_test, truth = self.get_batch(val_data)
        # Make predictions using the testing set
        diabetes_y_pred = self.regr.predict(diabetes_X_test)

        txt_test_result =""
        for index, s in enumerate(diabetes_y_pred):
            txt_test_result += "({},{:.2f})".format(index, s)
        txt_truth =""
        for index, s in enumerate(truth):
            txt_truth += "({},{:.2f})".format(index, s)

        metrics = Metrics()
        X_input = np.array(diabetes_X_test)[:,-1]
        metrics.performance_metrics("Bayes{}".format(self.code), X_input, diabetes_y_pred, truth)
        metrics.performance_values("Bayes{}".format(self.code), X_input, diabetes_y_pred, truth)

        plt.plot(diabetes_y_pred, color="red")
        plt.plot(truth, color="blue")
        plt.grid(True, which='both')
        plt.axhline(y=0, color='k')
        plt.savefig('graph/RandomForest-epoch.png')
        plt.close()

# A50 stocks
Dic.codes = sorted(Dic.codes,reverse=True)
for code in Dic.codes:
    # one stock
    model = MyRandomForest()
    model.code = code
    print("code={}".format(model.code))
    train_data, val_data = model.get_data(code)
    model.train(train_data)
    model.evaluate(val_data)
    model.plot_and_loss()