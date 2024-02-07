import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def load_data(path="data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    # 首先将文件中的内容读出，以二维数组的形式存储 2708
    # 以稀疏矩阵（采用CSR格式压缩）将数据中的特征存储 2708
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    # 对文献的编号构建字典(论文编号,索引值)
    # 读取cite文件，以二维数组的形式存储
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    # 生成图的边edges，（x,y）其中x、y都是为以文章编号为索引得到的值（也就是边对应的并非论文编号，而是字典中论文编号对应的索引值），
    # 此外，y中引入x的文献    
    # edges 是 为edges_unordered 统一分配序号，从0开始
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    # build symmetric adjacency matrix
    # 生成邻接矩阵，生成的矩阵为稀疏矩阵，对应的行和列坐标分别为边的两个点，该步骤之后得到的是一个有向图
    # edges是np.array数据，其中np.array.shape[0]表示行数，np.array.shape[1]表示列数
    # np.ones是生成全1的n维数组，第一个参数表示返回数组的大小
     # 无向图的领接矩阵是对称的，因此需要将上面得到的矩阵转换为对称的矩阵，从而得到无向图的领接矩阵
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)
    
    # 使用todense()方法将稀疏矩阵b转换成稠密矩阵c
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

# 该函数需要传入特征矩阵作为参数。对于本文使用的cora的数据集来说，每一行是一个样本，每一个样本是1433个特征。
# 归一化函数实现的方式：对传入特征矩阵的每一行分别求和，取到数后就是每一行非零元素归一化的值，然后与传入特征矩阵进行点乘。
# 其调用在第77行：features = normalize(features)
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1)) # 得到一个（2708,1）的矩阵
    r_inv = np.power(rowsum, -1).flatten() # 得到（2708，）的元组
    # 在计算倒数的时候存在一个问题，如果原来的值为0，则其倒数为无穷大，因此需要对r_inv中无穷大的值进行修正，更改为0
    # np.isinf()函数测试元素是正无穷还是负无穷
    r_inv[np.isinf(r_inv)] = 0.
    # 归一化后的稀疏矩阵
    r_mat_inv = sp.diags(r_inv)  # 构建对角元素为r_inv的对角矩阵
    # 用对角矩阵与原始矩阵的点积起到标准化的作用，原始矩阵中每一行元素都会与对应的r_inv相乘，最终相当于除以了sum
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)