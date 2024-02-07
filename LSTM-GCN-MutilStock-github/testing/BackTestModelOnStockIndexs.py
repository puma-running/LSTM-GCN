import time
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import sys, os
sys.path.append(os.path.abspath('.'))
from util.Dictionary import Dictionary as Dic
# '..' 表示当前所处的文件夹上一级文件夹的绝对路径；'.' 表示当前所处的文件夹的绝对路径
# from models.MLPModel import MLPModel as MLPModel
# from models.AlexNet import AlexNetModel as AlexNetModel
# from models.CarOneChanel_9301430 import Car
from models.Car import Car
from util.Ag_IndexsAccount import IndexsAccount as Account
# from util.Ag_IndexsAccountFan import IndexsAccount as AccountFan

from util.Log import Log
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import pandas as pd

# matplotlib.use('TkAgg')

def trainAndTest():
    '''
    模型的训练和测试
    nloop 循环次数
    '''
    # cates = ["open", "close"]
    cates = ["close"]
    models = [] 
    for cate in cates:    
        model = Car()
        model.initNet()
        model.loadtoloader()
        
        # 模型参数不存在，则训*+99练
        # PATH = 'pt/{}Model_{}_width1_{}_width2_{}.pt'.format(model.net.getModeTitle(), 'DCE.m2001', Dic.args.width1,  Dic.width2)
        if Dic.args.isMultiDataTrain == True:
            # 后Dic.with2中最大值
            # PATH = 'pt/{}_{}Model_{}_width1_{}_width2_{}-max.pt'.format(cate, model.net.getModeTitle(), code, Dic.width1,  Dic.width2)
            PATH = 'pt/{}_Model_{}_StockNum_{}_width_{}_{}_closedayes_{}_{}.pt'.format(cate, \
                model.net.__class__.__name__, len(Dic.codes), Dic.args.width1,  Dic.args.width2,\
                    Dic.args.closedayes, Dic.args.targetPoint)
            
            if os.path.exists(PATH) == True:
            # 模型装载
                if torch.cuda.is_available():
                    m_state_dict =  torch.load(PATH)
                    model.net.load_state_dict(m_state_dict)
                else:
                    # map_location='cpu' map_location=lambda storage, loc: storage
                    m_state_dict =  torch.load(PATH, map_location='cpu')
                    model.net.load_state_dict(m_state_dict)
            # 训练
            for epoch in range(Dic.args.epochs):
                correct1 = model.train(epoch)
                model.test(epoch)

            # torch.save(model.net.state_dict(), PATH)
            models.append(model)
        else:
            PATH = 'pt/{}_Model_{}_width1_{}_width2_{}.pt'.format(cate, model.net.getModeTitle(), Dic.width1,  Dic.width2)
            if os.path.exists(PATH) != True:
                # 训练
                for epoch in range(Dic.args.epochs):
                    correct1 = model.train(epoch)
                    model.test(epoch)
                # torch.save(model.net.state_dict(), PATH)
            else:
                # 模型装载
                if torch.cuda.is_available():
                    m_state_dict =  torch.load(PATH)
                    model.net.load_state_dict(m_state_dict)
                else:
                    # map_location='cpu' map_location=lambda storage, loc: storage
                    m_state_dict =  torch.load(PATH, map_location='cpu')
                    model.net.load_state_dict(m_state_dict)
            models.append(model)
    return models

def tradeOnTestdatas(models):
    '''
    在数据集上进行交易 # 交易回测
    '''
    # 交易回测
    closemodel = models[0]
    # openmodel = models[1]
    closemodel.net.eval()
    # openmodel.net.eval()
    Dic.start_time = Dic.k_times[Dic.test_begin].split('.')[0]
    Dic.end_time = Dic.k_times[Dic.test_end-1].split('.')[0]
    # _trends_allcodes, _inputs, _cur_times, _maxmin, _indexs = closemodel.test('tradeOnTestdatas')
    _trends_allcodes, _inputs, _cur_times, _maxmin, _indexs,  targets_res, outputs_res = closemodel.test('tradeOnTestdatas')
    # _, _trends_bid, _inputs_bid, _cur_times_bid, _maxmin_bid, _indexs_bid = openmodel.test('tradeOnTestdatas')
    
    _trends_allcodes = np.array(_trends_allcodes).transpose()
    inputs_res = np.array(_inputs).reshape(-1,35).transpose()
    targets_res = np.array(targets_res).reshape(-1,35).transpose()
    outputs_res = np.array(outputs_res).reshape(-1,35).transpose()
    txt_predict =""
    txt_truth =""
    for iCode in range(0, len(_trends_allcodes)):
        # 为论文收集数据
        code = Dic.codes[iCode]
        code = Dic.codes[iCode]
        closemodel.metrics.performance_metrics(code, inputs_res[iCode], outputs_res[iCode], targets_res[iCode])
        closemodel.metrics.performance_values(code, inputs_res[iCode], outputs_res[iCode], targets_res[iCode])

    for iCode in range(0, len(_trends_allcodes)):
        # 处理每个股票
        a1 = Account()
        a1.setInitValue(100000, 0)
        # a2 = AccountFan()
        # a2.setInitValue(100000, 0)
        # 重新设置股票代码
        Dic.code = Dic.codes[iCode]
        # if Dic.code not in Dic.codes_1:
        #     continue
        Dic.log.info("当前股票-{}".format(Dic.code))
        _trends =  _trends_allcodes[iCode]
        for i in range(0, len(_trends)):
            p_ask = _trends[i]
            # p_bid = _trends_bid[i]
            p_bid = _trends[i]

            arr = closemodel.getarr(_indexs[i], iCode)
            tupTime = time.localtime(_cur_times[i])#秒时间戳
            stadardTime = time.strftime("%Y-%m-%d %H:%M:%S", tupTime)
            # if predicted_new in [0,2]:
            #     Dic.log.info("{} 指标 {} 价格{}".format(stadardTime, f_cci, arr[4]))
            
            # predcited = random.choice([0, 1, 2])
            # a1.trading(p_bid, p_ask, arr)
            a1.trading(p_bid, p_ask, arr[0], arr[4])
            # 综合指标
            # if predcited in [0,2] and predicted_new != 1:
            #     account.trading(predicted_new, arr)
            # else:
            #     account.trading(predicted_new, arr)
        # 回测结束
        if len(_trends)!=0:
            arr = closemodel.getarr(_indexs[i], iCode)
            # a1.trading(88, 88, arr)
            a1.trading(88, 88, arr[0], arr[4])
            a1.printres()
            # if Dic.isOpenFan==1:
            #     a2.printres()

def visualTradingRusult():
    '''模拟交易结果可视化'''
    x_data = np.arange(0, len(Dic.balanceChange.values()))
    _values = []
    for i in sorted(Dic.balanceChange):
        _values.append(Dic.balanceChange[i])
    for i in  np.arange(0, len(_values)):
        if i!=0:
            _values[i] = _values[i]+_values[i-1]
    plt.plot(x_data, _values)
    # 存储图片
    log_path = './log'
    if os.path.exists(log_path) and os.path.isdir(log_path):
        pass
    else:
        os.mkdir(log_path)
    _now = time.strftime("%Y%m%d%H%M%S")
    _figname = os.path.join(log_path, "pic-{}.png".format(_now))
    plt.savefig(_figname)
    # plt.show()
    plt.cla() 
    Dic.log.info("总高成本{:.2f}".format(np.sum(Dic.costs_max)*100))
    Dic.log.info("总低成本{:.2f}".format(np.sum(Dic.costs_min)*100))
    Dic.log.info("总平均成本{:.2f}".format(np.sum(Dic.costs_mean)*100))
    Dic.log.info("总-手续费：{:.2f}".format(Dic.fees))
    Dic.log.info("总-收益：{:.2f}".format(Dic.profits))
    Dic.log.info("总-损失次数：{}".format(Dic.profitMinus))
    Dic.log.info("收-益次数：{}".format(Dic.profitPlus))
    
    result = dict()
    _benefits = dict()
    # 求出所有时间点
    for code in Dic.codes:
        if code in Dic.code_times.keys():
            times = Dic.code_times[code]
            indexs = Dic.code_indexs[code]
            short_long = Dic.code_short_long[code]
            for _time_open, _time_close, _indexs_open, _indexs_close, _short_long in zip(times[::2], times[1::2 ],indexs[::2], indexs[1::2 ], short_long):
                if _time_open in result.keys():
                    result[_time_open] += _indexs_open
                    _benefits[_time_open] += -float(_indexs_open)*Dic.args.nMultiplier*Dic.args.feeROpen*_short_long
                else:
                    result[_time_open] = _indexs_open
                    _benefits[_time_open] = -float(_indexs_open)*Dic.args.nMultiplier*Dic.args.feeROpen*_short_long
                
                if _time_close in result.keys():
                    result[_time_close] -= _indexs_close
                    _benefits[_time_close] += -float(_indexs_close)*Dic.args.nMultiplier*Dic.args.feeRClose*_short_long
                    _benefits[_time_close] += (float(_indexs_close)-float(_indexs_open))*Dic.args.nMultiplier*_short_long
                else:
                    result[_time_close] = -_indexs_close
                    _benefits[_time_close] = -float(_indexs_close)*Dic.args.nMultiplier*Dic.args.feeRClose*_short_long
                    _benefits[_time_close] += (float(_indexs_close)-float(_indexs_open))*Dic.args.nMultiplier*_short_long
    values = [0]
    for key in sorted(result):
        v = result[key]*Dic.args.nMultiplier+values[-1] - _benefits[key]
        values.append(v)
    Dic.log.info("总真实成本{:.2f}".format(np.max(values)))
    # 收益排序
    Dic.log.info("收益排序")
    Dic.log.info(sorted(Dic.profitpercode.items(), key = lambda kv:(kv[1], kv[0])))

    plt.plot(np.arange(0, len(result)+1), values)
    # 存储图片
    log_path = './log'
    if os.path.exists(log_path) and os.path.isdir(log_path):
        pass
    else:
        os.mkdir(log_path)
    _now = time.strftime("%Y%m%d%H%M%S")
    _figname = os.path.join(log_path, "pic-{}.png".format(_now))
    plt.savefig(_figname)
    # plt.show()
    plt.cla() 

    # 求每年收益 按照月
    _profits = dict()
    _peryear =dict()
    for code in Dic.codes:
        if code in Dic.code_times.keys():
            times = Dic.code_times[code]
            indexs = Dic.code_indexs[code]
            short_long = Dic.code_short_long[code]
            for _time_open, _time_close, _indexs_open, _indexs_close, _short_long in zip(times[::2], times[1::2 ],indexs[::2], indexs[1::2 ], short_long):
                if _time_close in _profits.keys():
                    _profits[_time_close] += (-float(_indexs_open)*Dic.args.nMultiplier*Dic.args.feeROpen-float(_indexs_close)*Dic.args.nMultiplier*Dic.args.feeRClose)*_short_long
                    _profits[_time_close] += ((float(_indexs_close)-float(_indexs_open))*Dic.args.nMultiplier)*_short_long
                else:
                    _profits[_time_close] = (-float(_indexs_open)*Dic.args.nMultiplier*Dic.args.feeROpen-float(_indexs_close)*Dic.args.nMultiplier*Dic.args.feeRClose)*_short_long
                    _profits[_time_close] += ((float(_indexs_close)-float(_indexs_open))*Dic.args.nMultiplier)*_short_long
    # 求每年收益 按照月
    for key in sorted(_profits):
        _strYear = key.split("-")[0]+"-"+key.split("-")[1]
        if _strYear in _peryear.keys():
            _peryear[_strYear] = round(_peryear[_strYear]+_profits[key],2)
        else:
            _peryear[_strYear] = round(_profits[key],2)
    Dic.log.info(_peryear)
    # 求每年收益 按照年
    _peryear =dict()
    for key in sorted(_profits):
        _strYear = key.split("-")[0]
        if _strYear in _peryear.keys():
            _peryear[_strYear] = round(_peryear[_strYear]+_profits[key],2)
        else:
            _peryear[_strYear] = round(_profits[key],2)
    Dic.log.info(_peryear)

# def spanByMonth():
#     '''获取一个股票数据的长度，其他数据和本数据对齐
#     把时间序列放到数组中self.k_times，然后跳出循环
#     '''
#     temp = list(set(Dic.codes))
#     temp.sort()
#     Dic.log.info(temp)
#     # 分割含头不含尾
#     span_month = []
#     code = Dic.codes[1]
#     # 股票数据路径
#     filename = Dic.filename.format(code)
#     if os.path.exists(filename) == False:
#         Dic.log.info("数据不存在{}".format(code))
#         return
#     Dic.log.info("数据装载{}".format(code))
#     df = pd.read_csv(filename)
#     use_cols = ["datetime","{}.open".format(code),"{}.high".format(code),"{}.low".format(code),"{}.close".format(code)
#         , "{}.volume".format(code),"{}.open_oi".format(code),"{}.close_oi".format(code)]
#     klines = df[use_cols]
#     Dic.k_times = np.array(klines[use_cols[0]])
#     Dic.log.info("股票{}，长度{}".format(code, len(Dic.k_times)))

#     nmonth_pre = '00'
#     # 含头不含尾
#     for index, value in enumerate(Dic.k_times):
#         nmonth = value[5:7]
#         if nmonth != nmonth_pre:
#             nmonth_pre = nmonth
#             span_month.append(int(index))
#     if span_month[-1] != len(Dic.k_times):
#         span_month.append(len(Dic.k_times))
#     return span_month

def loadData():
    '''
    装载数据到
    Dic.k_prices_open[code] Dic.k_prices_heigh[code] Dic.k_prices_low[code] Dic.k_prices_close[code]
    '''
    Dic.k_prices_open = dict()
    Dic.k_prices_heigh = dict()
    Dic.k_prices_low = dict()
    Dic.k_prices_close = dict()
    Dic.klines = dict()
    for code in Dic.codes:
        # 股票路径
        filename = Dic.filename.format(code)
        if os.path.exists(filename) == False:
            Dic.log.info("数据不存在{}".format(code))
            continue
        Dic.log.info("数据装载{}".format(code))
        df = pd.read_csv(filename)
        use_cols = ["datetime","{}.open".format(code),"{}.high".format(code),"{}.low".format(code),"{}.close".format(code)
            , "{}.volume".format(code),"{}.open_oi".format(code),"{}.close_oi".format(code)]
        klines = df[use_cols]
        Dic.klines[code] = df[use_cols]
        Dic.k_prices_open[code] = np.array(klines[use_cols[1]], dtype=np.float32)
        Dic.k_prices_heigh[code] = np.array(klines[use_cols[2]], dtype=np.float32)
        Dic.k_prices_low[code] = np.array(klines[use_cols[3]], dtype=np.float32)
        Dic.k_prices_close[code] = np.array(klines[use_cols[4]], dtype=np.float32)

if __name__ == "__main__":
    Dic.codes = sorted(Dic.codes,reverse=True)
    # Dic.codes = sorted(Dic.codes,reverse=False)
    Dic.initParameter(Dic.codes[0])
    Dic.initLog()
    loadData()
    # 分钟数据
    '''
    span_month = spanByMonth()
    for a, b, c in zip(span_month[0:-2],span_month[1:-1],span_month[2:]):
        Dic.point_a = a
        Dic.point_b = b
        Dic.point_c = c
        # 训练和测试 
        models = trainAndTest()
        # 交易回测
        tradeOnTestdatas(models)
        # 模拟交易结果可视化
        visualTradingRusult()
        # break
    '''
    # 日线数据
    Dic.k_times = np.array(Dic.klines[Dic.codes[2]]["datetime"])
    for i in range(22*10,len(Dic.k_times)-22*2, 22*2):
        # arrs_train.append(Dic.k_times[Dic.test_begin:Dic.test_end])
        # arrs_test.append(Dic.k_times[Dic.train_begin:Dic.train_end])
        Dic.log.info("当前月-{}-{}".format(Dic.k_times[i],Dic.k_times[i+22*2]))
        Dic.train_begin = 0
        # Dic.train_begin = i-22*10
        Dic.train_end = i
        # 5个1尺度
        Dic.test_begin = i-Dic.args.width1
        Dic.test_end = i+22*2
        # 训练和测试 
        models = trainAndTest()
        # 交易回测
        tradeOnTestdatas(models)
        # 模拟交易结果可视化
        visualTradingRusult()
    # 关闭日志 避免每个info重复输出
    Dic.log.closeLog()
    os.system("shutdown /s /t 0")
