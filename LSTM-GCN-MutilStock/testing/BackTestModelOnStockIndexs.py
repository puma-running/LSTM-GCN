import time
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import sys, os
sys.path.append(os.path.abspath('.'))
from util.Dictionary import Dictionary as Dic
from models.Car import Car
from util.Ag_IndexsAccount import IndexsAccount as Account

from util.Log import Log
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import pandas as pd

def trainAndTest():
    '''
    Model training and testing
    nloop Cycle number
    '''
    # cates = ["open", "close"]
    cates = ["close"]
    models = [] 
    for cate in cates:    
        model = Car()
        model.initNet()
        model.loadtoloader()
        if Dic.args.isMultiDataTrain == True:
            PATH = 'pt/{}_Model_{}_StockNum_{}_width_{}_{}_closedayes_{}_{}.pt'.format(cate, \
                model.net.__class__.__name__, len(Dic.codes), Dic.args.width1,  Dic.args.width2,\
                    Dic.args.closedayes, Dic.args.targetPoint)
            if os.path.exists(PATH) == True:
            # Model loading
                if torch.cuda.is_available():
                    m_state_dict =  torch.load(PATH)
                    model.net.load_state_dict(m_state_dict)
                else:
                    m_state_dict =  torch.load(PATH, map_location='cpu')
                    model.net.load_state_dict(m_state_dict)
            # training
            for epoch in range(Dic.args.epochs):
                correct1 = model.train(epoch)
                model.test(epoch)
            models.append(model)
        else:
            PATH = 'pt/{}_Model_{}_width1_{}_width2_{}.pt'.format(cate, model.net.getModeTitle(), Dic.width1,  Dic.width2)
            if os.path.exists(PATH) != True:
                for epoch in range(Dic.args.epochs):
                    correct1 = model.train(epoch)
                    model.test(epoch)
            else:
                if torch.cuda.is_available():
                    m_state_dict =  torch.load(PATH)
                    model.net.load_state_dict(m_state_dict)
                else:
                    m_state_dict =  torch.load(PATH, map_location='cpu')
                    model.net.load_state_dict(m_state_dict)
            models.append(model)
    return models

def tradeOnTestdatas(models):
    closemodel = models[0]
    # openmodel = models[1]
    closemodel.net.eval()
    # openmodel.net.eval()
    Dic.start_time = Dic.k_times[Dic.test_begin].split('.')[0]
    Dic.end_time = Dic.k_times[Dic.test_end-1].split('.')[0]
    _trends_allcodes, _inputs, _cur_times, _maxmin, _indexs,  targets_res, outputs_res = closemodel.test('tradeOnTestdatas')
    
    _trends_allcodes = np.array(_trends_allcodes).transpose()
    inputs_res = np.array(_inputs).reshape(-1,len(_trends_allcodes)).transpose()
    targets_res = np.array(targets_res).reshape(-1,len(_trends_allcodes)).transpose()
    outputs_res = np.array(outputs_res).reshape(-1,len(_trends_allcodes)).transpose()
    txt_predict =""
    txt_truth =""
    for iCode in range(0, len(_trends_allcodes)):
        # Collect data for the paper
        code = Dic.codes[iCode]
        code = Dic.codes[iCode]
        closemodel.metrics.performance_metrics(code, inputs_res[iCode], outputs_res[iCode], targets_res[iCode])
        closemodel.metrics.performance_values(code, inputs_res[iCode], outputs_res[iCode], targets_res[iCode])

    for iCode in range(0, len(_trends_allcodes)):
        a1 = Account()
        a1.setInitValue(100000, 0)
        Dic.code = Dic.codes[iCode]
        Dic.log.info("Current stock-{}".format(Dic.code))
        _trends =  _trends_allcodes[iCode]
        for i in range(0, len(_trends)):
            p_ask = _trends[i]
            p_bid = _trends[i]

            arr = closemodel.getarr(_indexs[i], iCode)
            tupTime = time.localtime(_cur_times[i])
            stadardTime = time.strftime("%Y-%m-%d %H:%M:%S", tupTime)
            a1.trading(p_bid, p_ask, arr[0], arr[4])
        # End of backtest
        if len(_trends)!=0:
            arr = closemodel.getarr(_indexs[i], iCode)
            a1.trading(88, 88, arr[0], arr[4])
            a1.printres()

def visualTradingRusult():
    '''Visualization of simulated trading results'''
    x_data = np.arange(0, len(Dic.balanceChange.values()))
    _values = []
    for i in sorted(Dic.balanceChange):
        _values.append(Dic.balanceChange[i])
    for i in  np.arange(0, len(_values)):
        if i!=0:
            _values[i] = _values[i]+_values[i-1]
    plt.plot(x_data, _values)
    log_path = './log'
    if os.path.exists(log_path) and os.path.isdir(log_path):
        pass
    else:
        os.mkdir(log_path)
    _now = time.strftime("%Y%m%d%H%M%S")
    _figname = os.path.join(log_path, "pic-{}.png".format(_now))
    # plt.savefig(_figname)
    # plt.show()
    plt.cla() 
    Dic.log.info("total high cost{:.2f}".format(np.sum(Dic.costs_max)*100))
    Dic.log.info("total low cost {:.2f}".format(np.sum(Dic.costs_min)*100))
    Dic.log.info("total average cost{:.2f}".format(np.sum(Dic.costs_mean)*100))
    Dic.log.info("Total-fees {:.2f}".format(Dic.fees))
    Dic.log.info("Total-earnings {:.2f}".format(Dic.profits))
    Dic.log.info("Total-loss times{}".format(Dic.profitMinus))
    Dic.log.info("Number of benefits{}".format(Dic.profitPlus))
    
    result = dict()
    _benefits = dict()
    # Find all the time points
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
    Dic.log.info("total true cost{:.2f}".format(np.max(values)))
    Dic.log.info("Income ranking")
    Dic.log.info(sorted(Dic.profitpercode.items(), key = lambda kv:(kv[1], kv[0])))

    plt.plot(np.arange(0, len(result)+1), values)
    log_path = './log'
    if os.path.exists(log_path) and os.path.isdir(log_path):
        pass
    else:
        os.mkdir(log_path)
    _now = time.strftime("%Y%m%d%H%M%S")
    _figname = os.path.join(log_path, "pic-{}.png".format(_now))
    plt.cla() 

    # annual income per month
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
    # annual income per month
    for key in sorted(_profits):
        _strYear = key.split("-")[0]+"-"+key.split("-")[1]
        if _strYear in _peryear.keys():
            _peryear[_strYear] = round(_peryear[_strYear]+_profits[key],2)
        else:
            _peryear[_strYear] = round(_profits[key],2)
    Dic.log.info(_peryear)
    # annual income per year
    _peryear =dict()
    for key in sorted(_profits):
        _strYear = key.split("-")[0]
        if _strYear in _peryear.keys():
            _peryear[_strYear] = round(_peryear[_strYear]+_profits[key],2)
        else:
            _peryear[_strYear] = round(_profits[key],2)
    Dic.log.info(_peryear)


def loadData():
    '''
    load data into
    Dic.k_prices_open[code] Dic.k_prices_heigh[code] Dic.k_prices_low[code] Dic.k_prices_close[code]
    '''
    Dic.k_prices_open = dict()
    Dic.k_prices_heigh = dict()
    Dic.k_prices_low = dict()
    Dic.k_prices_close = dict()
    Dic.klines = dict()
    for code in Dic.codes:
        # the path of the stock data 
        filename = Dic.filename.format(code)
        if os.path.exists(filename) == False:
            Dic.log.info("--No data exists{}".format(code))
            continue
        Dic.log.info("--Data loading{}".format(code))
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
    Dic.k_times = np.array(Dic.klines[Dic.codes[2]]["datetime"])
    for i in range(22*10,len(Dic.k_times)-22*2, 22*2):
        Dic.log.info("now month-{}-{}".format(Dic.k_times[i],Dic.k_times[i+22*2]))
        Dic.train_begin = 0
        Dic.train_end = i
        Dic.test_begin = i-Dic.args.width1
        Dic.test_end = i+22*2
        models = trainAndTest()
        tradeOnTestdatas(models)
        visualTradingRusult()
    Dic.log.closeLog()
    # os.system("shutdown /s /t 0")