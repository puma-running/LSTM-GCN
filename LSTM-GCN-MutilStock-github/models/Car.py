
import time
import numpy as np
import copy
# import talib as ta
import math
import sys, os
sys.path.append(os.path.abspath('.'))
# from util.Tradition import Tradition as Tradition
from util.Dictionary import Dictionary as Dic
import random

import argparse
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import torch.optim as optim
from pygcn.models import GCN
from util.Metrics import Metrics
# from PyEMD import EMD

class Car():
    """准备数据，模型训练，模型测试"""
    def __init__(self):

        """ 刚开始训练时：学习率以 0.01 ~ 0.001 为宜。
        一定轮数过后：逐渐减缓。
        接近训练结束：学习速率的衰减应该在100倍以上。
        """
        #上级目录。 '..' 表示当前所处的文件夹上一级文件夹的绝对路径；'.' 表示当前所处的文件夹的绝对路径
        # self.loadtoloader()
        #trainloader 放在训练中，每次都 shuffle
        # self.tradition = Tradition()
        # self.emd = EMD()
        self.metrics = Metrics()

    def loadtoloader(self):
        '''
        数据准备:数据集分为两份，一份训练，一份测试
        Dic.point是训练结束点 是测试开始点
        Dic.point_end是测试结束点
        '''
        arrs_train =[]
        arrs_test =[]
        codes = []
        arrs_train.append(Dic.k_times[Dic.train_begin:Dic.train_end])
        arrs_test.append(Dic.k_times[Dic.test_begin:Dic.test_end])
        for code in Dic.codes:
            use_cols = ["datetime","{}.open".format(code),"{}.high".format(code),"{}.low".format(code),"{}.close".format(code)
                , "{}.volume".format(code),"{}.open_oi".format(code),"{}.close_oi".format(code)]
            klines = Dic.klines[code]

            # 判断时间序列是否对应
            temp = np.array(klines[use_cols[0]])
            bequal = (temp==Dic.k_times)
            # 判断那个股票数据最多，就是没有停盘
            if len(temp)!=len(Dic.k_times) or bequal.all()==False:
                Dic.log.info("数据长度不够的股票代码{}".format(code))
                Dic.errCodes.append(code)
                continue
            else:
                codes.append(code)
            # for v in [use_cols[4], use_cols[2], use_cols[3], use_cols[5]]:
            for v in [use_cols[4], use_cols[5]]:
                # 训练样本集其他数据
                arrs_train.append(np.array(klines[v][Dic.train_begin:Dic.train_end], dtype=np.float32))
                # 测试样本集其他数据
                arrs_test.append(np.array(klines[v][Dic.test_begin:Dic.test_end], dtype=np.float32))
        # 归一化 # 准备样本集  # 0,self.nwidth1-1为输入样本  # 因为买入时间和卖出时间的间隔，即预测间隔 # 输入长度+间隔 为目标
        self.inputs1, self.fTargets1, self.maxmin1, self.cur_times1, self.indexs1 = self.__getSamplesFromOnecode(arrs_train)
        self.inputs2, self.fTargets2, self.maxmin2, self.cur_times2, self.indexs2 = self.__getSamplesFromOnecode(arrs_test)
        # 装入dataset， Dataloader
        trainset = Data.TensorDataset(torch.tensor(self.inputs1), torch.tensor(self.fTargets1), torch.tensor(self.maxmin1), torch.tensor(self.cur_times1), torch.tensor(self.indexs1))
        # 装载器 nbatch_size=16的时候召回很差，要不正向预测号 要不负向预测号
        nbatch_size = 100
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=nbatch_size, shuffle=True)
        testset = Data.TensorDataset(torch.tensor(self.inputs2), torch.tensor(self.fTargets2), torch.tensor(self.maxmin2), torch.tensor(self.cur_times2), torch.tensor(self.indexs2))
        # 装载器
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
        # 修订使用到的codes
        for tt in Dic.errCodes:
            if tt in Dic.codes:
                Dic.codes.remove(tt)
        temp = list(set(Dic.codes))
        temp.sort()
        Dic.log.info(temp)
        
    def initNet(self):
        '''初始化神经网络模型'''
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.criterion = nn.MSELoss()
        # self.criterion = nn.L1Loss()
        self.criterion = nn.SmoothL1Loss()
        self.entroy=nn.CrossEntropyLoss()
        # self.net = GcnTrain()
        # self.optimizer = torch.optim.Adam(self.net.parameters(), lr=Dic.LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0) 
        
        # 多GPU运行
        # if torch.cuda.device_count() > 1:
        #     print("Use", torch.cuda.device_count(), 'gpus')
        #     self.net = nn.DataParallel(self.net)
        # 单GPU运行
        # self.net = self.net.to(self.device)
        # cpu运行
        # net = net.to('cpu')

        # Training settings
        parser = argparse.ArgumentParser()
        """模型结构参数"""
        # parser.add_argument('--no-cuda', action='store_true', default=False,
        #                     help='Disables CUDA training.')
        # parser.add_argument('--fastmode', action='store_true', default=False,
        #                     help='Validate during training pass.')

        parser.add_argument('--lstm_in_dim', type=int, default=2)
        parser.add_argument('--lstm_hidden_dim', type=int, default=1)
        # parser.add_argument('--windows_week', type=int, default=5)
        
        parser.add_argument('--lstm_layers', type=int, default=1)
        # parser.add_argument('--gc1_in_dim', type=int, default=Dic.args.width1+Dic.args.width1_week,
        parser.add_argument('--gc1_in_dim', type=int, default=Dic.args.width1,
                            help='Number of input units.') 
        # parser.add_argument('--gc1_hidden_dim', type=int, default=1,
        parser.add_argument('--gc1_hidden_dim', type=int, default=35,
                            help='Number of hidden units.')  
        # parser.add_argument('--gc1_hidden_dim', type=int, default=int(Dic.args.width1/2),
        #                     help='Number of hidden units.')
        parser.add_argument('--nclass', type=int, default=1) # 买 卖 持有，分类问题 3  # 回归问题 1

        """# 训练参数"""
        parser.add_argument('--seed', type=int, default=42, help='Random seed.')
        parser.add_argument('--weight_decay', type=float, default=5e-4,
                            help='Weight decay (L2 loss on parameters).')
        parser.add_argument('--lr', type=float, default=0.01,
                            help='Initial learning rate.')
        # parser.add_argument('--cuda', default=(not args.no_cuda and torch.cuda.is_available()))
        parser.add_argument('--cuda', default=torch.cuda.is_available())
        parser.add_argument('--dropout', type=float, default=0.5,
                            help='Dropout rate (1 - keep probability).')
        parser.add_argument('--device', default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        # lstm_in_dim, lstm_hidden_dim, lstm_layers = 2, 128, 1
        args = parser.parse_args()
        Dic.log.info(args)
        # args.cuda = not args.no_cuda and torch.cuda.is_available()
        # args.features = Dic.width1*2

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed) 

        # Load data
        # adj, features, labels, idx_train, idx_val, idx_test = load_data()

        # Model and optimizer
        # model = GCN(nfeat=features.shape[1],
        #             nhid=args.hidden,
        #             nclass=labels.max().item() + 1,
        #             dropout=args.dropout)

        model = GCN(args)
        optimizer = optim.Adam(model.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)
        if args.cuda:
            model.cuda()
            # features = features.cuda()
            # adj = adj.cuda()
            # labels = labels.cuda()
            # idx_train = idx_train.cuda()
            # idx_val = idx_val.cuda()
            # idx_test = idx_test.cuda()

        # 变量传递
        self.args = args
        # self.adj, self.features, self.labels, self.idx_train, self.idx_val, self.idx_test \
        #     = adj, features, labels, idx_train, idx_val, idx_test
        self.model, self.optimizer = model, optimizer
        self.net = model

    def __list_split(self,items, n, span):
        '''将一个数组，均分为多个数组，n为样本长度，span为预测间隔，items为输入序列'''
        return [items[i:i+n] for i in range(0, len(items)-span-n+1)]

    def __getSamplesFromOnecode(self, arrs):
        '''
        arrs是矩阵 每行为一个股票 如果4个特征 例如 open height low close volume
        则有这个股票的4行 nFeatures=4
        width1是输入窗口的尺寸
        width2是Minute-Interval Prediction
        那么第一个input数组的最后一个元素的索引应该是Dic.width1-1
        那么最后一个input数组的最后一个元素的索引应该是len(arrs[0])-nwidth2-1
        '''
        nFeatures = 2
        nwidth1, nwidth2 = Dic.args.width1, Dic.args.width2
        inputs, targets, maxmin, cur_times, indexs = [], [], [], [], []
        
        begin = -1
        for i in range(Dic.args.width1, len(arrs[0])-nwidth2):
            # print(i)
            strtime = arrs[0][i][0:19]
            strptime = time.strptime(strtime,"%Y-%m-%d %H:%M:%S")
            mktime = int(time.mktime(strptime))
            #数据分段
            # current_datetime = arrs[0][i][0:19]
            # current_time = current_datetime.split(" ")[1].split(".")[0]
            if i%60000==0:
                print(i)
            
            # 1.时间序列
            # strtime = arrs[vIndex][i+nwidth1-1][0:19]
            # strptime = time.strptime(strtime,"%Y-%m-%d %H:%M:%S")
            # mktime = int(time.mktime(strptime))
            # cur_times.append(mktime)
            # temp1 = arrs[1][i]
            # temp2 = arrs[2][i]
            one_target = []
            one_maxmin = []
            one_input = []
            # 处理所有股票的数据，形成通道
            begin = i-nwidth1+1
            # if begin >= 0:
            # 只训练某个时间
            # if begin >= 0 and current_time =='14:30:00':
            # 训练所有时间
            if begin >= 0:
                for kk in range(1, len(arrs), nFeatures):
                    _features =[]
                    # 价格-open 价格-high 价格-low 成交量
                    '''归一化0-1'''
                    for iF in range(nFeatures):
                        # 尺度1，按天
                        arr1 = arrs[kk+iF][begin:i+1]

                        pmax, pmin, eps = max(arr1), min(arr1), 0
                        if pmax == pmin:
                            eps = 1e-5
                        arr1 = (arr1-pmin)/(pmax-pmin+eps)
                        # 标签
                        if iF in [0]: #价格-close
                            target = arrs[kk+iF][i+nwidth2]
                            target = (target-pmin)/(pmax-pmin+eps)
                            one_target.append(target)
                            one_maxmin.append([pmax,pmin])
                        # _features.append([res1,res2])
                        # _features.append([arr1,arr1])
                        _features.append(arr1)
                    # 添加时间标注

                    one_input.append(_features)
                # 记录输入、预测真实值、归一化、位置、当前时间等
                # inputs.append([arr1, arr2])
                # targets.append([target])
                # maxmin.append([pmax,pmin])
                inputs.append(one_input)
                targets.append(one_target)
                maxmin.append(one_maxmin)
                indexs.append(i)
                cur_times.append(mktime)
                # elif '09:30:00'<current_time <'14:31:00' and len(arr1)>0 and len(arr2)>0:
                    # arr1.append(temp1)
                    # arr2.append(temp2)
        return np.array(inputs, dtype=np.float32), np.array(targets,dtype=np.float32), np.array(maxmin,dtype=np.float32), np.array(cur_times), np.array(indexs)

    def shuffleTrain(self):
        '''打乱两个数组'''
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
        # 装载器 nbatch_size=16的时候召回很差，要不正向预测号 要不负向预测号
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
        # self.shuffleTrain()
        print('\nEpoch: %d' % epoch)
        self.net.train()
        trends, inputs_res, targets_res, outputs_res = [], [], [], []
        # _metrics = Metrics()
        
        for batch_idx, (inputs, targets, maxmin, cur_times, indexs) in enumerate(self.trainloader):
            # 1.Forward
            regularization_loss = 0
            loss_regression = 0
            loss_classification = 0
            self.optimizer.zero_grad()
            # outputs = self.net(self.formatInputs(inputs))
            outputs = []
            _targets = targets.to(self.device)
            # 处理min-batch中每个元素
            for features, _target in zip(inputs,_targets):
                _output = self.net(features)
                # 训练
                # loss += self.criterion(_output, _target) + 0.1*regularization_loss
                loss_regression += self.myloss_regression(features, _output.reshape(-1), _target) + 0.1*regularization_loss
                # loss_classification += self.myloss_classification(features[:,-1], _output.reshape(-1), _target) + 0.1*regularization_loss
                outputs.append(np.array(_output.to('cpu').detach()).reshape(-1).tolist())

            # 2. Backward
            loss_regression.backward(retain_graph=True)
            # 3. Update
            # self.optimizer.step()
            # self.optimizer.zero_grad()
            # loss_classification.backward()
            self.optimizer.step()

            # self.optimizer.zero_grad()
            # (self.net.a2*loss_regression + self.net.a1*loss_classification).backward()
            # self.optimizer.step()

            _trends, _inputs, _targets, _outputs = self.toTrends(inputs, targets, outputs, maxmin)
            trends += _trends
            inputs_res += _inputs
            targets_res += _targets
            outputs_res += _outputs
        # 格式变化，然后性能评价
        # 格式变化，然后性能评价
        self.metrics.performance_metrics("train", inputs_res, outputs_res, targets_res)

    def test(self, epoch):
        '''
        在数据集上进行测试
        '''
        # global best_acc
        self.net.eval()
        cur_times_res, maxmin_res, indexs_res = [], [], []
        trends_res, inputs_res, targets_res, outputs_res = [], [], [], []
        # _metrics = Metrics()
        with torch.no_grad():
            for batch_idx,  (inputs, targets, maxmins, cur_times, indexs) in enumerate(self.testloader):
                
                regularization_loss = 0
                loss = 0
                outputs = []
                for features, _target in zip(inputs, targets.to(self.device)):
                    # _feature, _adj = load_data_stock(features, Dic.args.codes)
                    # df=pd.DataFrame(_adj)
                    # _adj = df.fillna(0).values
                    # _adj = torch.FloatTensor(_adj)
                    # _feature = torch.FloatTensor(_feature)
                    # _feature, _adj = _feature.to(self.device), _adj.to(self.device)
                    _output = self.net(features)
                    # 训练
                    # loss += self.criterion(_output, _target) + 0.1*regularization_loss
                    loss += self.myloss_regression(features, _output.reshape(-1), _target) + 0.1*regularization_loss
                    # 追加到列表
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
        return trends_res, inputs_res, cur_times_res, maxmin_res, indexs_res, targets_res, outputs_res

    def getarr(self, index, iCode):
        """ index为数据的最后分钟线 """
        # 借鉴
        index = int(index)
        current_time = Dic.k_times[Dic.test_begin:][index] # 输入序列的最后一个时间为当前时间 \
        
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
        返回趋势，返回判断预测的准确情况
        '''
        _inputs, _targets, _outputs = [], [], []
        trends =[]
        for x1Item, x2Item, x3Item, mmItem in zip(inputs, fTargets, outputs, maxmin):
            # 单尺度
            # x1Item = x1Item[::,0,-1] #例如为  [0.3303]  准确率高的原因？ x1[-1][-3]
            # 多尺度
            x1Item = x1Item[::,0,-1] #例如为  [0.3303]  准确率高的原因？ x1[-1][-3]
            _trends =[]
            for x1, x2, x3, max_min in zip(x1Item, x2Item, x3Item, mmItem):
                fMax_min = max_min[0]-max_min[1]
                '''规划到0-1'''
                x1, x2, x3 = x1*fMax_min+max_min[1], x2*fMax_min+max_min[1], x3*fMax_min+max_min[1]
                '''规划到0-2'''
                # x1, x2, x3 = (x1-1)*fMax_min+max_min[2], (x2-1)*fMax_min+max_min[2], (x3-1)*fMax_min+max_min[2]
                _inputs.append(x1)
                _targets.append(x2)
                _outputs.append(x3)

                npoint = x1*Dic.args.targetPoint
                """ 预测情况 """
                if x3-x1 >= npoint:
                    _trends.append(2)
                elif x3-x1 <= -npoint:
                    _trends.append(0)
                else:
                    _trends.append(1)
            trends.append(_trends)

        return trends, _inputs, _targets, _outputs

    # 自己写的误差
    def myloss_regression(self, features, _output, _target):
        loss = self.criterion(_output, _target)
        return loss
        loss = 0
        for x1, x2, x3 in zip(features, _target, _output):
            x1, x2, x3 = x1[-1], x2, x3[0]
            loss += abs(torch.exp(-(x3-x2)*(x2-x1)) - 1)
        return loss
    
    # 自己写的误差
    def myloss_classification(self, features, _output, _target):
        # loss = self.criterion(outputs, targets.to(self.device)) + regularization_loss # MSELoss

        # loss = nn.CrossEntropyLoss()
        # input = torch.randn(3, 5, requires_grad=True)
        # target = torch.empty(3, dtype=torch.long).random_(5)
        # output = loss(input, target)
        # output.backward()
        
        # # Example of target with class probabilities
        # input = torch.randn(3, 5, requires_grad=True)
        # target = torch.randn(3, 5).softmax(dim=1)
        # output = loss(input, target)
        # output.backward()

        # class_output = []
        # class_target = []
        loss = 0
        for i_f, i_o, i_t in zip(features, _output, _target):
            oneLoss = (i_o-i_f)/abs(i_o-i_f) - (i_t-i_f)/abs(i_t-i_f)
            if torch.isnan(oneLoss) == False:
                loss += abs(oneLoss)
            # class_output.append( (i_o-i_f)/abs(i_o-i_f) )
            # class_target.append( (i_t-i_f)/abs(i_t-i_f) )
        # class_output = (_output-features)/abs(_output-features)
        # class_target = (_target-features)/abs(_target-features)
        # loss = self.entroy(class_output, class_target)
        # loss = self.criterion(np.array(class_output), np.array(class_target))
        return loss
        loss = 0
        for x1, x2, x3 in zip(features, _target, _output):
            x1, x2, x3 = x1[-1], x2, x3[0]
            loss += abs(torch.exp(-(x3-x2)*(x2-x1)) - 1)
        return loss
