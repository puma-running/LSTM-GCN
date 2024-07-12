
import time
import numpy as np
import copy
import math
import sys, os
sys.path.append(os.path.abspath('.'))
from util.Tradition import Tradition as Tradition
from util.Dictionary import Dictionary as Dic
import random

import argparse
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import torch.optim as optim
from baselineTrade.TSRM.modelsTSRM import GCN
from util.Metrics import Metrics
from PyEMD import EMD
from models.ModelHook import My_hook_lstm, My_hook_GCN, My_hook_linear, Hook_process

class Car():
    def __init__(self):
        self.emd = EMD()
        self.metrics = Metrics()

    def loadtoloader(self):
        arrs_train =[]
        arrs_test =[]
        codes = []
        arrs_train.append(Dic.k_times[Dic.train_begin:Dic.train_end])
        arrs_test.append(Dic.k_times[Dic.test_begin:Dic.test_end])
        for code in Dic.codes:
            use_cols = ["datetime","{}.open".format(code),"{}.high".format(code),"{}.low".format(code),"{}.close".format(code)
                , "{}.volume".format(code),"{}.open_oi".format(code),"{}.close_oi".format(code)]
            klines = Dic.klines[code]

            temp = np.array(klines[use_cols[0]])
            bequal = (temp==Dic.k_times)
            if len(temp)!=len(Dic.k_times) or bequal.all()==False:
                Dic.log.info("数据长度不够的股票代码{}".format(code))
                Dic.errCodes.append(code)
                continue
            else:
                codes.append(code)
            for v in [use_cols[1], use_cols[4], use_cols[2], use_cols[3], use_cols[5]]:
                arrs_train.append(np.array(klines[v][Dic.train_begin:Dic.train_end], dtype=np.float32))
                arrs_test.append(np.array(klines[v][Dic.test_begin:Dic.test_end], dtype=np.float32))
        self.inputs1, self.fTargets1, self.maxmin1, self.cur_times1, self.indexs1 = self.__getSamplesFromOnecode(arrs_train)
        self.inputs2, self.fTargets2, self.maxmin2, self.cur_times2, self.indexs2 = self.__getSamplesFromOnecode(arrs_test)
        # Load the dataset and dataloader.
        trainset = Data.TensorDataset(torch.tensor(self.inputs1), torch.tensor(self.fTargets1), torch.tensor(self.maxmin1), torch.tensor(self.cur_times1), torch.tensor(self.indexs1))
        nbatch_size = 100
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=nbatch_size, shuffle=True)
        testset = Data.TensorDataset(torch.tensor(self.inputs2), torch.tensor(self.fTargets2), torch.tensor(self.maxmin2), torch.tensor(self.cur_times2), torch.tensor(self.indexs2))
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
        for tt in Dic.errCodes:
            if tt in Dic.codes:
                Dic.codes.remove(tt)
        temp = list(set(Dic.codes))
        temp.sort()
        Dic.log.info(temp)
        
    def initNet(self):
        '''初始化神经网络模型'''
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.SmoothL1Loss()
        self.entroy=nn.CrossEntropyLoss()
        
        # Training settings
        parser = argparse.ArgumentParser()

        parser.add_argument('--lstm_in_dim', type=int, default=2)
        parser.add_argument('--lstm_hidden_dim', type=int, default=1)
        
        parser.add_argument('--lstm_layers', type=int, default=2)
        parser.add_argument('--gc1_in_dim', type=int, default=Dic.args.width1,
                            help='Number of input units.') 
        parser.add_argument('--gc1_hidden_dim', type=int, default=35,
                            help='Number of hidden units.')  
        parser.add_argument('--nclass', type=int, default=1) # 买 卖 持有，分类问题 3  # 回归问题 1

        parser.add_argument('--seed', type=int, default=42, help='Random seed.')
        parser.add_argument('--weight_decay', type=float, default=5e-4,
                            help='Weight decay (L2 loss on parameters).')
        parser.add_argument('--lr', type=float, default=0.01,
                            help='Initial learning rate.')
        parser.add_argument('--cuda', default=torch.cuda.is_available())
        parser.add_argument('--dropout', type=float, default=0.2,
                            help='Dropout rate (1 - keep probability).')
        parser.add_argument('--device', default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        args = parser.parse_args()
        Dic.log.info(args)

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed) 

        model = GCN(args)
        optimizer = optim.Adam(model.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)
        if args.cuda:
            model.cuda()

        self.args = args
        self.model, self.optimizer = model, optimizer
        self.net = model

    def __list_split(self,items, n, span):
        return [items[i:i+n] for i in range(0, len(items)-span-n+1)]

    def __getSamplesFromOnecode(self, arrs):
        nFeatures = 5
        nwidth1, nwidth2 = Dic.args.width1, Dic.args.width2
        inputs, targets, maxmin, cur_times, indexs = [], [], [], [], []
        
        begin = -1
        for i in range(nwidth1, len(arrs[0])-nwidth2):
            strtime = arrs[0][i][0:19]
            strptime = time.strptime(strtime,"%Y-%m-%d %H:%M:%S")
            mktime = int(time.mktime(strptime))
            if i%60000==0:
                print(i)
            
            one_target = []
            one_maxmin = []
            one_input = []
            begin = i-nwidth1+1
            if begin >= 0:
                for kk in range(1, len(arrs), nFeatures):
                    _features =[]
                    for iF in range(nFeatures):
                        arr1 = arrs[kk+iF][begin:i+1]

                        pmax, pmin, eps = max(arr1), min(arr1), 0
                        if pmax == pmin:
                            eps = 1e-5
                        arr1 = (arr1-pmin)/(pmax-pmin+eps)
                        if iF in [1]: 
                            target = arrs[kk+iF][i+nwidth2]
                            target = (target-pmin)/(pmax-pmin+eps)
                            one_target.append(target)
                            one_maxmin.append([pmax,pmin])
                        _features.append(arr1)
                    one_input.append(_features)
                inputs.append(one_input)
                targets.append(one_target)
                maxmin.append(one_maxmin)
                indexs.append(i)
                cur_times.append(mktime)
        return np.array(inputs, dtype=np.float32), np.array(targets,dtype=np.float32), np.array(maxmin,dtype=np.float32), np.array(cur_times), np.array(indexs)

    def shuffleTrain(self):
        state = np.random.get_state()
        np.random.shuffle(self.inputs1)
        np.random.set_state(state)
        np.random.shuffle(self.fTargets1)
        np.random.set_state(state)
        np.random.shuffle(self.maxmin1)
        np.random.set_state(state)
        np.random.shuffle(self.cur_times1)
        np.random.set_state(state)
        np.random.shuffle(self.indexs1)

        trainset = Data.TensorDataset(torch.tensor(self.inputs1), torch.tensor(self.fTargets1), torch.tensor(self.maxmin1), torch.tensor(self.cur_times1), torch.tensor(self.indexs1))
        nbatch_size = 100
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=nbatch_size, shuffle=True)

    def flast_price(self, x):
        return x[-1]
    def fopen_price(self, x):
        return x[0]
    def f_sub(self, x):
        return x[-1]-x[0]
        
    def train(self, epoch):
        '''
        # Training
        '''
        print('\nEpoch: %d' % epoch)
        self.net.train()
        trends, inputs_res, targets_res, outputs_res = [], [], [], []
        
        for batch_idx, (inputs, targets, maxmin, cur_times, indexs) in enumerate(self.trainloader):
            # 1.Forward
            regularization_loss = 0
            loss_regression = 0
            loss_classification = 0
            self.optimizer.zero_grad()
            outputs = []
            _targets = targets.to(self.device)
            for features, _target in zip(inputs,_targets):
                _output = self.net(features)
                loss_regression += self.myloss_regression(features, _output.reshape(-1), _target) + 0.1*regularization_loss
                outputs.append(np.array(_output.to('cpu').detach()).reshape(-1).tolist())

            # 2. Backward
            loss_regression.backward(retain_graph=True)
            # 3. Update
            self.optimizer.step()

            _trends, _inputs, _targets, _outputs = self.toTrends(inputs, targets, outputs, maxmin)
            trends += _trends
            inputs_res += _inputs
            targets_res += _targets
            outputs_res += _outputs
        self.metrics.performance_metrics("train", inputs_res, outputs_res, targets_res)

    def test(self, epoch):
        # global best_acc
        self.net.eval()
        cur_times_res, maxmin_res, indexs_res = [], [], []
        trends_res, inputs_res, targets_res, outputs_res = [], [], [], []
        with torch.no_grad():
            # register_forward_hoo
            my_hook_lstm = My_hook_lstm()
            self.net.lstm1.register_forward_hook(my_hook_lstm.forward_hook)
            my_hook_GCN1 = My_hook_GCN()
            self.net.gc1.register_forward_hook(my_hook_GCN1.forward_hook)
            my_hook_GCN2 = My_hook_GCN()
            self.net.gc1.register_forward_hook(my_hook_GCN2.forward_hook)
            my_hook_linear = My_hook_linear()
            self.net.classifier.register_forward_hook(my_hook_linear.forward_hook) 

            for batch_idx,  (inputs, targets, maxmins, cur_times, indexs) in enumerate(self.testloader):
                regularization_loss = 0
                loss = 0
                outputs = []
                for features, _target in zip(inputs, targets.to(self.device)):
                    _output = self.net(features)
                    loss += self.myloss_regression(features, _output.reshape(-1), _target) + 0.1*regularization_loss
                    # Add to the list
                    outputs.append(np.array(_output.to('cpu').detach()).reshape(-1).tolist())
                    if np.isnan(loss.to('cpu')):
                        print(batch_idx)

                trends, inputs, targets, outputs = self.toTrends(inputs, targets, outputs, maxmins)
                trends_res += trends
                inputs_res += inputs
                targets_res += targets
                outputs_res += outputs

                cur_times_res += cur_times
                maxmin_res += maxmins
                indexs_res += indexs
            self.metrics.performance_metrics("test", inputs_res, outputs_res, targets_res)
            # anlysis hook
            _hook_process = Hook_process()
            _hook_process.process_hook(my_hook_lstm, my_hook_GCN1, my_hook_GCN1, my_hook_linear)
        return trends_res, inputs_res, cur_times_res, maxmin_res, indexs_res, targets_res, outputs_res

    def getarr(self, index, iCode):
        # The "index" represents the last minute line of data.
        index = int(index)
        current_time = Dic.k_times[Dic.test_begin:][index] 
        # The last time of the input sequence is the current time.
        
        _beginDay = index-Dic.args.width1+1
        
        price_open = Dic.k_prices_open[Dic.codes[iCode]][Dic.test_begin:][_beginDay]
        price_heigh = np.max(Dic.k_prices_heigh[Dic.codes[iCode]][Dic.test_begin:][_beginDay:index])
        price_low = np.min(Dic.k_prices_low[Dic.codes[iCode]][Dic.test_begin:][_beginDay:index])
        price_close = Dic.k_prices_close[Dic.codes[iCode]][Dic.test_begin:][index]

        arr=[]
        arr.append(current_time)
        arr.append(price_open)
        arr.append(price_heigh)
        arr.append(price_low)
        arr.append(price_close)
        return arr

    def toTrends(self, inputs, fTargets, outputs, maxmin):
        '''
        Return to the trend, and see how accurate the predictions were.
        '''
        _inputs, _targets, _outputs = [], [], []
        trends =[]
        for x1Item, x2Item, x3Item, mmItem in zip(inputs, fTargets, outputs, maxmin):
            # Single Scale
            # Multi-scale
            x1Item = x1Item[::,1,-1]
            _trends =[]
            for x1, x2, x3, max_min in zip(x1Item, x2Item, x3Item, mmItem):
                fMax_min = max_min[0]-max_min[1]
                '''Planning to 0-1'''
                x1, x2, x3 = x1*fMax_min+max_min[1],\
                x2*fMax_min+max_min[1], x3*fMax_min+max_min[1]
                '''Planning to 0-2'''
                _inputs.append(x1)
                _targets.append(x2)
                _outputs.append(x3)

                npoint = x1*Dic.args.targetPoint
                if x3-x1 >= npoint:
                    _trends.append(2)
                elif x3-x1 <= -npoint:
                    _trends.append(0)
                else:
                    _trends.append(1)
            trends.append(_trends)

        return trends, _inputs, _targets, _outputs

    def myloss_regression(self, features, _output, _target):
        loss = self.criterion(_output, _target)
        return loss
    
    def myloss_classification(self, features, _output, _target):
        loss = 0
        for i_f, i_o, i_t in zip(features, _output, _target):
            oneLoss = (i_o-i_f)/abs(i_o-i_f) - (i_t-i_f)/abs(i_t-i_f)
            if torch.isnan(oneLoss) == False:
                loss += abs(oneLoss)
        return loss