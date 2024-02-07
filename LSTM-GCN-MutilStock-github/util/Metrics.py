import sys, os
sys.path.append(os.path.abspath('.'))
from util.Dictionary import Dictionary as Dic

import numpy as np
class Metrics():
    # PERFORMANCE EVALUATION OF
    # 3. 混淆矩阵 other following metrics, 
    # (TP for true positive, TN for true negative, FP for false positive, and FN for false negative):
    Dic.code='baselinemethod'
    Dic.initLog()
    def confusionMatrix(self, inputs, outputs, targets):
        _TP, _FP, _TN, _FN = 0, 0, 0, 0
        for _i1, _o1, _t1 in zip(inputs, outputs, targets):
            # npoint = _i1*Dic.targetPoint
            npoint = 0
            """ 预测情况 """
            if _o1-_i1 > npoint and _t1-_i1 > npoint:
                _TP += 1
            elif _o1-_i1 > npoint and _t1-_i1 <= npoint:
                _FP += 1
            elif _o1-_i1 <= npoint and _t1-_i1 <= npoint:
                _TN += 1
            elif _o1-_i1 <= npoint and _t1-_i1 > npoint:
                _FN += 1
        return _TP, _FP, _TN, _FN
    # For the binary price movement prediction, we use area
    # under roc curve (AUC) and other following metrics, (TP for
    # true positive, TN for true negative, FP for false positive, and
    # FN for false negative):
    # 2. Hit Rates 命中率
    # Accuracy表征的是预测正确的样本比例。不过通常不用这个概念，主要是因为预测正确的负样本这个没有太大意义。
    def myAccuracy(self, inputs, outputs, targets):
        TP, FP, TN, FN = self.confusionMatrix(inputs, outputs, targets)
        if (TP+FP+TN+FN)!=0:
            acc = (TP+TN)/(TP+FP+TN+FN)
        else:
            acc = 0
        return acc

    # Precision：查准率 正向样本
    # Precision表征的是预测正确的正样本的准确度，查准率等于预测正确的正样本数量/所有预测为正样本数量。
    # Precision越大说明误检的越少，Precision越小说明误检的越多。
    def myPrecision_pos(self, inputs, outputs, targets):
        TP, FP, TN, FN = self.confusionMatrix(inputs, outputs, targets)
        if (TP+FP)!=0:
            precision = (TP)/(TP+FP)
        else:
            precision=0
        return precision

    def myPrecision_neg(self, inputs, outputs, targets):
        TP, FP, TN, FN = self.confusionMatrix(inputs, outputs, targets)
        if (TN+FN)!=0:
            precision = (TN)/(TN+FN)
        else:
            precision=0
        return precision

    # Recall：查全率
    # Recall表征的是预测正确的正样本的覆盖率，查全率等于预测正确的正样本数量/所有正样本的总和，
    # TP+TN实际就是Ground Truth的数量。Recall越大说明漏检的越少，Recall越小说明漏检的越多。
    def myRecall_pos(self, inputs, outputs, targets):
        TP, FP, TN, FN = self.confusionMatrix(inputs, outputs, targets)
        if (TP+FN)!=0:
            recall = (TP)/(TP+FN)
        else:
            recall = 0
        return recall

    def myRecall_neg(self, inputs, outputs, targets):
        TP, FP, TN, FN = self.confusionMatrix(inputs, outputs, targets)
        if (TN+FP)!=0:
            recall = (TN)/(TN+FP)
        else:
            recall = 0
        return recall

    def myF1_pos(self, inputs, outputs, targets):
        p1 = self.myPrecision_pos(inputs, outputs, targets)
        r1 = self.myRecall_pos(inputs, outputs, targets)
        if p1+r1!=0:
            F1 = 2*p1*r1/(p1+r1)
        else:
            F1 = 0
        return F1

    def myF1_neg(self, inputs, outputs, targets):
        p1 = self.myPrecision_neg(inputs, outputs, targets)
        r1 = self.myRecall_neg(inputs, outputs, targets)
        if p1+r1!=0:
            F1 = 2*p1*r1/(p1+r1)
        else:
            F1 = 0
        return F1
    # 1. we also considered the different loss value
    # mean squared error (MSE)
    def MAE(self, outputs, targets):
        metric = np.sum(np.abs(outputs-targets))
        metric = metric/len(targets)
        return metric

    # mean absolute error (MAE)
    def MSE(self, outputs, targets):
        metric = np.sum(np.square(outputs-targets))
        metric = metric/len(targets)
        return metric

    def _percentage_error(self, actual: np.ndarray, predicted: np.ndarray):
        """ Percentage error """
        return (actual - predicted) / actual

   # Mean absolute value percentage error (MAPE)
    def MAPE(self, predicted, actual):
        return np.mean(np.abs(self._percentage_error(actual, predicted)))

    def mdape(self, actual: np.ndarray, predicted: np.ndarray):
        """
        11 中位数绝对误差百分比 MedAPE
        Median Absolute Percentage Error
        Note: result is NOT multiplied by 100
        """
        return np.median(np.abs(self._percentage_error(actual, predicted)))
    
    def performance_metrics(self, str, inputs_res, outputs_res, targets_res):
        metrics = Metrics()
        acc = metrics.myAccuracy(inputs_res, outputs_res, targets_res)*100  
        pre_pos = metrics.myPrecision_pos(inputs_res, outputs_res, targets_res)*100  
        pre_neg = metrics.myPrecision_neg(inputs_res, outputs_res, targets_res)*100  
        rec_pos = metrics.myRecall_pos(inputs_res, outputs_res, targets_res)*100  
        rec_neg = metrics.myRecall_neg(inputs_res, outputs_res, targets_res)*100  
        F1_pos = metrics.myF1_pos(inputs_res, outputs_res, targets_res)*100  
        F1_neg = metrics.myF1_neg(inputs_res, outputs_res, targets_res)*100  
        mae = metrics.MAE(np.array(outputs_res), np.array(targets_res))
        mse = metrics.MSE(np.array(targets_res), np.array(outputs_res))
        Dic.log.warning('code:{} mae:{:.4f} mse:{:.4f} acc:{:.2f} pre_pos:{:.2f} pre_neg:{:.2f} rec_pos:{:.2f} rec_neg:{:.2f} F1_pos:{:.2f} F1_neg:{:.2f} '\
            .format(str, mae, mse, acc, pre_pos, pre_neg, rec_pos, rec_neg, F1_pos, F1_neg))
        Dic.log.csv_performance_metrics('{} {:.4f} {:.4f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}' 
            .format(str, mae, mse, acc, pre_pos, pre_neg, rec_pos, rec_neg, F1_pos, F1_neg)) 
    
    def performance_values(self, str, inputs_res, outputs_res, targets_res):
        txt_inputs =""
        txt_outputs =""
        txt_targets =""
        # if code =='SSE.600031':
        for index, s in enumerate(inputs_res):
            txt_inputs += "({},{:.2f})".format(index, s)
        for index, s in enumerate(outputs_res):
            txt_outputs += "({},{:.2f})".format(index, s)
        for index, s in enumerate(targets_res):
            txt_targets += "({},{:.2f})".format(index, s)
        Dic.log.csv_performance_values(str, txt_inputs, txt_outputs, txt_targets)