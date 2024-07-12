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
from PyEMD import EMD

#KMeans
from sklearn.cluster import KMeans

class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.args = args
        self.lstm1 = nn.LSTM(5, 64, 2,\
            dropout= args.dropout)
        self.lstm2 = nn.LSTM(64, 128, 2,\
            dropout= args.dropout)

        gc1 = GraphConvolution(128, 128)

        classifier = nn.Linear(128, args.nclass)

        self.gc1 = gc1.to(self.args.device)

        self.classifier = classifier.to(self.args.device)

        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)

    def forward(self, features):
        '''Call for single scale or multi-scale'''
        return self.forward_Onescale(features)

    def forward_Onescale(self, features):
        # Feature Extraction - Scale 1
        x = features.permute(2,0,1)
        x = x.to(self.args.device)
        out, (h_n1, c_n) = self.lstm1(x)
        out, (h_n2, c_n) = self.lstm2(out)
        x = h_n2[1, :, :]

        # Create the feature matrix and adjacency matrix.
        closeprices = features[:, 0, :]
        dad = self.compuyterDAD(closeprices)
        df=pd.DataFrame(dad)
        dad = df.fillna(0).values
        dad = torch.FloatTensor(dad)
        dad = dad.to(self.args.device)

        # Graph convolutional computation
        x = F.relu(self.gc1(x, dad))
        x = F.dropout(x, self.args.dropout, training=self.training)
        x = self.classifier(x)
        return x

    def forward_Multiscale(self, features):
        # Feature Extraction - Scale 1
        xs = []
        for i in range(2):
            x = features[:,:,i,:]
            x = x.permute(2,0,1)
            x = x.to(self.args.device)
            out, (h_n, c_n) = self.lstms[i](x)
            x = out[:, :, 0].permute(1,0)
            xs.append(x)
        x = torch.cat((xs[0], xs[1]), 1)
        # Create the feature matrix and adjacency matrix.
        dad = self.compuyterDAD(x)
        df=pd.DataFrame(dad)
        dad = df.fillna(0).values
        dad = torch.FloatTensor(dad)
        dad = dad.to(self.args.device)

        # Graph convolutional computation
        x = F.relu(self.gc1(x, dad))
        x = F.dropout(x, self.args.dropout, training=self.training)
        return x

    def compuyterDAD(self, features):
        """
        pearson spearman kendall
        Load citation network dataset (cora only for now)
        s://blog.csdn.net/small__roc/article/details/123519616
        """
        kmeans = KMeans(n_clusters=10)  # divide into three clusters
        # 10 clusters
        # cluster
        kmeans.fit(features.tolist())
        # Get the clusters center
        kmeans.cluster_centers_
        # Gets the cluster label for each sample
        labels = kmeans.labels_
        # An all-0 matrix with 35 rows and 35 columns
        adj = np.zeros((35, 35))
        for i in range(0,35):
            for j in range(0,35):
                if labels[i] == labels[j]:
                    adj[i][j] = 1
                
        newAdj = torch.as_tensor(adj.astype(np.float32))
        degreeMx = torch.eye(len(newAdj))
        for i in range(len(newAdj)):
            for j in range(len(newAdj)):
                if i ==j:
                    degreeMx[i][j] = torch.pow(torch.sum(newAdj[i]), -1/2)

        dad = torch.matmul(torch.matmul(degreeMx, newAdj), degreeMx)
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
        """
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
        degreeMx = torch.eye(len(newAdj))
        for i in range(len(newAdj)):
            for j in range(len(newAdj)):
                if i ==j:
                    degreeMx[i][j] = torch.pow(torch.sum(newAdj[i]), -1/2)

        dad = torch.matmul(torch.matmul(degreeMx, newAdj), degreeMx)
        return dad