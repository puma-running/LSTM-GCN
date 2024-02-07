import torch
import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution
import numpy as np
import sys, os
sys.path.append(os.path.abspath('.'))
from util.Dictionary import Dictionary as Dic
from pygcn.utils import normalize
import pandas as pd
# from PyEMD import EMD

class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.args = args
        # 尺度一
        self.lstm1 = nn.LSTM(args.lstm_in_dim, args.lstm_hidden_dim, args.lstm_layers,\
            dropout= args.dropout)
        # lstm2 = nn.LSTM(args.lstm_in_dim, args.lstm_hidden_dim, args.lstm_layers,\
        #     dropout= args.dropout)
        # lstm1 = lstm1.to(self.args.device)
        # lstm2 = lstm2.to(self.args.device)

        gc1 = GraphConvolution(args.gc1_in_dim, args.gc1_hidden_dim)
        gc2 = GraphConvolution(args.gc1_hidden_dim, args.gc1_hidden_dim)
        classifier = nn.Linear(args.gc1_hidden_dim, args.nclass)
        # classifier1 = nn.Linear(35*35, 35*17)
        # classifier2 = nn.Linear(35*17, 35)
        self.gc1 = gc1.to(self.args.device)
        self.gc2 = gc2.to(self.args.device)
        self.classifier = classifier.to(self.args.device)
        # self.classifier1 = classifier1.to(self.args.device)
        # self.classifier2 = classifier2.to(self.args.device)
        # self.lstms = [lstm1, lstm2]
        # self.lstms = [lstm1]
        # self.linear_correlation = nn.Linear(Dic.args.width1, Dic.args.width1)
        # self.gc1 = GraphConvolution(args.gc1_in_dim, args.gc1_hidden_dim)
        # self.classifier = nn.Linear(args.gc1_hidden_dim, args.nclass)
        # self.gc2 = GraphConvolution(args.gc1_hidden_dim, args.nclass)
        # self.a1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.a2 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # 新加的参数
        # adj = torch.empty(len(Dic.codes), len(Dic.codes))
        # adj = nn.init.uniform_(adj)
        # self.adj = nn.Parameter(torch.FloatTensor(adj), requires_grad=True)
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)

    def forward(self, features):
        '''调用单尺度或者多尺度'''
        return self.forward_Onescale(features)
        # return self.forward_Multiscale(features)

    def forward_Onescale(self, features):
        # 特征提取 尺度一
        # x_features = features[:,:,0,:]
        x = features.permute(2,0,1)
        x = x.to(self.args.device)
        out, (h_n, c_n) = self.lstm1(x)
        # 此时可以从out中获得最终输出的状态h
        x = out[:, :, 0].permute(1,0)

        # xlinear = torch.reshape(x_features.permute(0,2,1),(35,-1))
        # xlinear = x_features.to(self.args.device)
        # xlinear= self.linear_correlation(xlinear)
        # x = xlinear
        # x = torch.cat((x,xlinear),-1)
        # x = out.permute(1,0,2).reshape(35,24)
        # x = h_n[-1, :, :]
        # x = self.classifier(x)
        # 构建特征矩阵和邻接矩阵
        dad = self.compuyterDAD(x)
        # dad = self.compuyterDAD_bylinear(x)
        df=pd.DataFrame(dad)
        dad = df.fillna(0).values
        dad = torch.FloatTensor(dad)
        # x = torch.FloatTensor(x)
        dad = dad.to(self.args.device)

        # 图卷积计算
        x = F.relu(self.gc1(x, dad))
        x = F.relu(self.gc2(x, dad))
        x = F.dropout(x, self.args.dropout, training=self.training)
        # x = x.reshape(-1)
        x = self.classifier(x)
        # x = self.classifier1(x)
        # x = self.classifier2(x)
        # x = self.gc2(x, dad)
        # return F.log_softmax(x, dim=1)
        return x

    def forward_Multiscale(self, features):
        # 特征提取 尺度一
        xs = []
        for i in range(2):
            x = features[:,:,i,:]
            x = x.permute(2,0,1)
            x = x.to(self.args.device)
            out, (h_n, c_n) = self.lstms[i](x)
            # 此时可以从out中获得最终输出的状态h
            x = out[:, :, 0].permute(1,0)
            xs.append(x)
            # x = h_n[-1, :, :]
        x = torch.cat((xs[0], xs[1]), 1)
        # 构建特征矩阵和邻接矩阵
        dad = self.compuyterDAD(x)
        # dad = self.compuyterDAD_bylinear(x)
        df=pd.DataFrame(dad)
        dad = df.fillna(0).values
        dad = torch.FloatTensor(dad)
        dad = dad.to(self.args.device)

        # 图卷积计算
        x = F.relu(self.gc1(x, dad))
        x = F.dropout(x, self.args.dropout, training=self.training)
        # x = self.classifier(x)
        # x = self.gc2(x, dad)
        # return F.log_softmax(x, dim=1)
        return x

    def compuyterDAD(self, features):
        """
        pearson spearman kendall
        Load citation network dataset (cora only for now)
        s://blog.csdn.net/small__roc/article/details/123519616
        计算协方差，度矩阵
        计算协方差"""
        # n=len(arr)
        adj = []
        for i in range(len(features)):
            X = features[i]
            _cor = []
            for j in range(len(features)):
                Y = features[j]
                rhoXY = self.pearsonr(X,Y)
                # rhoXY = self.spearmanr(X,Y)
                _cor.append(rhoXY)
            adj.append(_cor)
        newAdj = torch.as_tensor(adj) + torch.eye(len(adj))
        # 计算度矩阵
        degreeMx = torch.eye(len(newAdj))
        for i in range(len(newAdj)):
            for j in range(len(newAdj)):
                if i ==j:
                    degreeMx[i][j] = torch.pow(torch.sum(newAdj[i]), -1/2)

        dad = torch.matmul(torch.matmul(degreeMx, newAdj), degreeMx)

        # features = normalize(features)
        # adj = normalize(adj + sp.eye(adj.shape[0]))
        # dad = normalize(dad)
        return dad

    def pearsonr(self, X, Y):
        XY = X*Y
        EX = X.mean()
        EY = Y.mean()
        EX2 = (X**2).mean()
        EY2 = (Y**2).mean()
        EXY = XY.mean()
        numerator = EXY - EX*EY                                 # 分子
        denominator = torch.sqrt(EX2-EX**2)*torch.sqrt(EY2-EY**2) # 分母
        if denominator == 0:
            return 'NaN'
        rhoXY = numerator/denominator
        return rhoXY

    def spearmanr(self,X,Y):
        EX = X.mean()
        EY = Y.mean()
        numerator = torch.sum((X-EX)*(Y-EY))
        denominator = torch.sqrt(torch.sum(torch.pow(X-EX,2))*torch.sum(torch.pow(Y-EY,2)))
        rhoXY = numerator/denominator
        return rhoXY

    def compuyterDAD_bylinear(self, features):
        """
        pearson，spearman，kendall
        Load citation network dataset (cora only for now)
        s://blog.csdn.net/small__roc/article/details/123519616
        计算协方差，度矩阵
        计算协方差"""
        # n=len(arr)
        adj = []
        for i in range(len(features)):
            X = features[i]
            _cor = []
            for j in range(len(features)):
                Y = features[j]
                temp = torch.cat([X, Y], 0)
                rhoXY = self.linear_correlation(temp)
                _cor.append(rhoXY[0])
            adj.append(_cor)
        newAdj = torch.as_tensor(adj) + torch.eye(len(adj))
        # 计算度矩阵
        degreeMx = torch.eye(len(newAdj))
        for i in range(len(newAdj)):
            for j in range(len(newAdj)):
                if i ==j:
                    degreeMx[i][j] = torch.pow(torch.sum(newAdj[i]), -1/2)

        dad = torch.matmul(torch.matmul(degreeMx, newAdj), degreeMx)

        # features = normalize(features)
        # adj = normalize(adj + sp.eye(adj.shape[0]))
        # dad = normalize(dad)
        return dad