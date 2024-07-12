import pandas as pd
import numpy as np
import math
import time
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import datetime
from contextlib import closing
import sys,os
sys.path.append(os.path.abspath('.'))
import logging
from util.Log import Log
from util.Dictionary import Dictionary as Dic

class IndexsAccount():
    '''
    code Indicates the index code of the account
    start_time Start time
    end_time End time format '2020-04-20 09:30:00','2020-12-04 14:59:00')
    feerate = 18/10000
    one 300 yuan for each index
    10% margin
    '''
    def __init__(self):
        self.nminutes = 0
        self.nsuccess = 0 
        self.nfail =0

        self.position_open = [] 
        self.balance = []

        # Transaction data
        self.sTimes = []            #the times of open and close position
        self.lTimes = []            #the times of open and close position
        self.sIndexs = []           #the index of open and close position
        self.lIndexs = []           #the index of open and close position

        # Current position
        self.nLong =0
        self.nShort =0
        self.acum = 0

        # Forecast interval (in klines), 
        # used in the account for stop loss and clearance time judgment,
        self.profitMinus = 0
        self.profitPlus = 0
        self.position_open_short =[]
        self.position_open_long = []
        # Holding time
        self.timsspan_longopen = 0
        self.timsspan_shortopen = 0
        self.nClose = 0
        self.openprice_long = 0
        self.openprice_short = 0
        self.code =""

    def setInitValue(self, hhBalance, hhPos):
        self.balance.append(hhBalance)
        self.pos = hhPos
    
    def __getwithdrawalrate(self):
        '''Get the maximum retracement rate'''
        withdrawalrates =(np.array(self.balance)[::2]-self.balance[0])/self.balance[0]
        minwithdrawalrate = min(withdrawalrates)
        return minwithdrawalrate

    def _accBlances(self):
        '''Get the total return curve
        and the total cost'''
        times = self.sTimes+self.lTimes
        indexs = self.sIndexs+self.lIndexs
        Dic.code_times[Dic.code] = times
        Dic.code_indexs[Dic.code] = indexs
        Dic.code_short_long[Dic.code] = self.position_open_short + self.position_open_long

        _poss = self.position_open_short + self.position_open_long
        arrays = []
        for _time_open, _time_close, _indexs_open, _indexs_close, _pos\
            in zip(times[::2], times[1::2 ],indexs[::2], indexs[1::2 ], _poss):
            arrays.append([_time_open, _time_close, _indexs_open, _indexs_close, _pos])
            # Cumulative income
            aFee = float(_indexs_open)*Dic.args.nMultiplier*Dic.args.feeROpen* abs(float(_pos)) + float(_indexs_close)*Dic.args.nMultiplier*Dic.args.feeRClose*abs(float(_pos))
            aBenefit = (float(_indexs_close)-float(_indexs_open))*Dic.args.nMultiplier*float(_pos)
            if _time_close in Dic.balanceChange.keys():
                Dic.balanceChange[_time_close] += aBenefit-aFee
            else:
                Dic.balanceChange[_time_close] = aBenefit-aFee
        # get the total cost
        if len(indexs)>0:
            Dic.costs_max.append(round(np.max(indexs[::2]),2))
            Dic.costs_min.append(round(np.min(indexs[::2]),2))
            Dic.costs_mean.append(round(np.mean(indexs[::2]),2))

    def __draw_profits(self):
        times = self.sTimes+self.lTimes
        indexs = self.sIndexs+self.lIndexs
        _poss = self.position_open_short + self.position_open_long

        x_data = np.arange(0,int(len(indexs)/2))
        arrays = []
        _profits = []
        _fees = []
        _times = []
        fprofit = 0 
        ffee = 0
        for _time_open, _time_close, _indexs_open, _indexs_close, _pos\
            in zip(times[::2], times[1::2 ],indexs[::2], indexs[1::2 ], _poss):
            arrays.append([_time_open, _time_close, _indexs_open, _indexs_close, _pos])
        a = np.array(arrays)[np.lexsort(np.array(arrays)[:,::-1].T)]

        for _time_open, _time_close, _indexs_open, _indexs_close, _pos\
            in zip(a[:,0], a[:,1], a[:,2], a[:,3], a[:,4]):
            fprofit += (float(_indexs_close)-float(_indexs_open))*Dic.args.nMultiplier*float(_pos)
            _profits.append( fprofit )
            if "RM" in Dic.code:
                ffee += Dic.args.feeROpen* abs(float(_pos)) + Dic.args.feeRClose*abs(float(_pos))
            else:
                ffee += float(_indexs_open)*Dic.args.nMultiplier*Dic.args.feeROpen* abs(float(_pos)) + float(_indexs_close)*Dic.args.nMultiplier*Dic.args.feeRClose*abs(float(_pos))
            _fees.append( ffee)     #手续费
            _times.append(_time_close)

        # 定义figure
        plt.figure()
        # 分隔figure
        gs = gridspec.GridSpec(5, 3)
        ax1 = plt.subplot(gs[0:3, :])
        ax2 = plt.subplot(gs[3, :])
        ax3 = plt.subplot(gs[4, :])

        ax1.plot(x_data, np.array(_profits)-np.array(_fees))
        ax2.plot(x_data, indexs[::2])
        ax3.plot(x_data, np.array(indexs[1::2])-np.array(indexs[::2]))
        # plt.show()
        print("draw")

    def printres(self):
        # Dic.log.info("交易情况：")
        # Dic.log.info(self.position_open)
        # Dic.log.info("开仓时间：")
        # Dic.log.info(self.time_open) 
        # Dic.log.info("清仓时间：")
        # Dic.log.info(self.time_close) 
        # Dic.log.info("收益情况：")
        # Dic.log.info(self.profits)
        # Dic.log.info("手续费情况：")
        # Dic.log.info(self.fees)
        times = self.sTimes+self.lTimes
        indexs = self.sIndexs+self.lIndexs
        _poss = self.position_open_short + self.position_open_long

        Dic.log.info("==按天统计=================")
        # self.__draw_profits()
        if len(indexs)>0:
            oneday_profits = 0 
            _begin = '1990-12-16 08:00:00.000000000'
            # _poss = [-1]*int(len(self.sTimes)/2)+[1]*int(len(self.lTimes)/2)
            arrays = []
            for _time_open, _time_close, _indexs_open, _indexs_close, _pos\
                in zip(times[::2], times[1::2 ],indexs[::2], indexs[1::2 ], _poss):
                arrays.append([_time_open, _time_close, _indexs_open, _indexs_close, _pos])
            a = np.array(arrays)[np.lexsort(np.array(arrays)[:,::-1].T)]

            for _time_open, _time_close, _indexs_open, _indexs_close, _pos\
                in zip(a[:,0], a[:,1], a[:,2], a[:,3], a[:,4]):
                
                _profits = (float(_indexs_close)-float(_indexs_open))*Dic.args.nMultiplier*float(_pos)

                _subday, _subminitues = self.Caltime(_begin.split(".")[0], _time_open.split(".")[0])
                if _subday >=1:
                    #新的一天　输出前一天的合计
                    Dic.log.info("{:.2f}".format(oneday_profits))
                    Dic.log.info("－－－－－－－－")
                    oneday_profits = _profits
                    _begin = _time_open.split(".")[0].split(" ")[0]+" 08:00:00.000000000"
                else:
                    oneday_profits += _profits

                Dic.log.info("{:.2f},{},{},{},{},{:.2f}".format(float(_indexs_open), _indexs_close, _time_open, _time_close, _pos, _profits))
                
            # 最后一天
            Dic.log.info("{:.2f}".format(oneday_profits))
        # === 统计结果
        Dic.log.info("===汇总统计==========")
        Dic.log.info("账户余额：{:.2f}".format(self.balance[-1]))
        ntimes = len(_poss)
        Dic.log.info("交易次数：{}".format(ntimes))

        _fees = 0
        for _open, _close, _pos in zip(indexs[::2], indexs[1::2], _poss):
            if "RM" in Dic.code:
                _fees += Dic.args.feeROpen* abs(_pos)     #手续费
                _fees += Dic.args.feeRClose*abs(_pos)     #手续费
            else:
                _fees += _open*Dic.args.nMultiplier*Dic.args.feeROpen* abs(_pos)     #手续费
                _fees += _close*Dic.args.nMultiplier*Dic.args.feeRClose*abs(_pos)     #手续费

        Dic.log.info("手续费：{:.2f}".format(_fees))

        _profits = 0
        for index1, index2, pos in zip(indexs[::2], indexs[1::2], _poss):
            _profits += (index2 - index1)*(pos) * Dic.args.nMultiplier
        Dic.log.info("{} 收益：{:.2f}".format(Dic.code, _profits))
        Dic.profitpercode[Dic.code]= round(_profits-_fees,2)
        
        if ntimes!=0:
            Dic.log.info("胜率{:.2f}".format(self.nsuccess/ntimes))
            Dic.log.info("失败率{:.2f}".format(self.nfail/ntimes))
            Dic.log.info("最大回撤率：{:.4f}".format(self.__getwithdrawalrate()))
        Dic.log.info("{} 损失次数：{}".format(Dic.code, self.profitMinus))
        Dic.log.info("{} 收益次数：{}".format(Dic.code, self.profitPlus))

        # 多数据集统计
        Dic.profitMinus += self.profitMinus
        Dic.profitPlus += self.profitPlus
        Dic.profits += _profits
        Dic.fees +=_fees
        Dic.log.info("总-手续费：{:.2f}".format(Dic.fees))
        Dic.log.info("总-收益：{:.2f}".format(Dic.profits))
        Dic.log.info("总-损失次数：{}".format(Dic.profitMinus))
        Dic.log.info("收-益次数：{}".format(Dic.profitPlus))


        #交易现金流情况
        # self.__draw_profits()
        self._accBlances()
        self.balance = []     #账户约的变动集合，初始余额1万
        # self.profits =[]              #清仓或止损每次的收益集合
        # self.fees = []                #手续费集合
        # self.margins = []             #股指保证金缴纳集合          #股指保证金缴纳集合


    #计算两个日期相差分钟数，自定义函数名，和两个日期的变量名。
    def Caltime(self, date1,date2):
        '''
        date2 - date1
        返回天，分钟差
        2020-04-20 09:30:00','2020-12-04 14:59:00')
        #%Y-%m-%d为日期格式，其中的-可以用其他代替或者不写，但是要统一，同理后面的时分秒也一样；可以只计算日期，不计算时间。
        '''
        date1=time.strptime(date1,"%Y-%m-%d %H:%M:%S") 
        date2=time.strptime(date2,"%Y-%m-%d %H:%M:%S")
        #根据上面需要计算日期还是日期时间，来确定需要几个数组段。下标0表示年，小标1表示月，依次类推...
        date1=datetime.datetime(date1[0],date1[1],date1[2],date1[3],date1[4],date1[5])
        date2=datetime.datetime(date2[0],date2[1],date2[2],date2[3],date2[4],date2[5])
        # date1=datetime.datetime(date1[0],date1[1],date1[2])
        # date2=datetime.datetime(date2[0],date2[1],date2[2])
        #返回两个变量相差的值，就是相差天数
        #print((date2-date1).days)#将天数转成int型
        return (date2-date1).days, (date2-date1).seconds/60

    def longOpen(self):
        '''
        #判断当天趋势的方向。若t时刻股指高于股指开盘价，则预测当日趋势为向上，在市场平稳的情况下可以做多，
        '''
        _open = self.ask_price+Dic.args.robber_open
        current_time = self.current_time
        # if self.lessthan(self.price_low_1, _open, self.price_heigh_1):
        if 1==1:
            # 记录开仓后至今一共多少K线
            self.acum = 0

            # 反向账户
            # if self.a2.nShort==0 and Dic.isOpenFan==1:
            #     self.a2.shortOpen()

            # Dic.log.info("当前位置：{}".format(self.kindex))
            self.openprice_long = _open

            self.lTimes.append(current_time)     #交易时间
            self.lIndexs.append(_open)  #交易价格        
            # self.pos = int(self.balance[-1]/(_open*self.nMultiplier*self.margin+_open*self.nMultiplier*self.feeROpen)
            #     *self.volume_rate)
            # if self.pos==0:
            #     print("错误")
            # self.nLong = Dic.numOpen_stock
            self.nLong = int(Dic.args.cashPerStock/_open/100)
            Dic.log.info("买多于{} 价格{:.2f} 数量{}".format(current_time, _open, self.nLong))
            self.position_open_long.append(self.nLong)     #修改当前仓位
            _margins = _open*Dic.args.nMultiplier*Dic.args.margin*abs(self.nLong)           #
            if "RM" in Dic.code:
                _fees = Dic.args.feeROpen*abs(self.nLong)     #缴纳手续费
            else:
                _fees = _open*Dic.args.nMultiplier*Dic.args.feeROpen*abs(self.nLong)     #缴纳手续费

            self.balance.append(self.balance[-1] - _margins- _fees)      #账户余额变动
            self.timsspan_longopen = 0
            return True
        return False
    def shortOpen(self):
        '''
        #判断当天趋势的方向。若t时刻股指高于股指开盘价，则预测当日趋势为向上，在市场平稳的情况下可以做多，
        '''
        # Dic.log.info("当前位置：{}".format(self.kindex))
        current_time = self.current_time
        _open = self.bid_price-Dic.robber_open
        
        # if self.lessthan(self.price_low_1, _open, self.price_heigh_1):
        if 1==1:
            # 
            if self.a2.nLong==0 and Dic.isOpenFan==1:
                self.a2.longOpen()

            # self.level = _open
            self.openprice_short = _open
            self.acum = 0  # 记录开仓后至今一共多少K线

            self.sTimes.append(current_time)     #交易时间
            self.sIndexs.append(_open)  #交易价格        
            # self.pos = int(self.balance[-1]/(_open*self.nMultiplier*self.margin+_open*self.nMultiplier*self.feeROpen)
            #     *self.volume_rate)
            # self.nShort = -Dic.numOpen_stock
            self.nShort = -int(Dic.args.cashPerStock/_open/100)
            Dic.log.info("卖空于{} 价格{:.2f} 数量{}".format(current_time, _open, self.nShort))
            #买空为负
            # self.pos = -self.pos
            self.position_open_short.append(self.nShort)     #修改当前仓位
            _margins = _open*Dic.args.nMultiplier*Dic.args.margin*abs(self.nShort)           #
            if "RM" in Dic.code:
                _fees = Dic.args.feeROpen*abs(self.nShort)     #缴纳手续费
            else:
                _fees = _open*Dic.args.nMultiplier*Dic.args.feeROpen*abs(self.nShort)     #缴纳手续费

            self.balance.append(self.balance[-1] - _margins- _fees)      #账户余额变动
            self.timsspan_shortopen = 0

            return True
        return False

    #回测结束清仓离场 #股指平仓、清仓
    def close_position(self, _category):
        '''
        每个指数点self.nMultiplier元
        '''
        # Dic.log.info("当前位置：{}".format(self.kindex))
        current_time = self.current_time
        
        _open = self.bid_price
        # if _open <=self.openprice_long and self.p_bid!=88 and self.p_ask!=88:
        #     _open = self.openprice_long+Dic.robber_close

        # if _category =='long' and self.lessthan(self.price_low_1, _open, self.price_heigh_1):
        if _category =='long':
            # 反向账户
            # if self.a2.nShort!=0 and Dic.isOpenFan==1:
            #     self.a2.close_position('short')

            if _open-self.openprice_long>0:
                self.nsuccess += 1
                self.profitPlus +=1
                self.nClose += 1
                restr ="收益"
                Dic.log.info("平多收益 {} {:.2f} {} {:.2f} {:.2f} {} {}".format(\
                    self.lTimes[-1][0:19], self.openprice_long, current_time[0:19], _open, \
                    _open-self.openprice_long, self.nLong, round((_open-self.openprice_long)*self.nLong*100,2)))
            else:
                self.nfail += 1
                self.profitMinus += 1
                self.nClose += 1
                restr ="止损"
                Dic.log.info("平多止损 {} {:.2f} {} {:.2f} {:.2f} {} {}".format(\
                    self.lTimes[-1][0:19], self.openprice_long, current_time[0:19], _open, \
                    _open-self.openprice_long, self.nLong, round((_open-self.openprice_long)*self.nLong*100,2)))
            self.acum = 0  # 记录开仓后至今一共多少K线
            self.lIndexs.append(_open) #交易价格
            if "RM" in Dic.code:
                _fees = Dic.args.feeRClose*abs(self.nLong)     #缴纳手续费
            else:
                _fees = _open*Dic.args.nMultiplier*Dic.args.feeRClose*abs(self.nLong)     #缴纳手续费

            _profits = (_open-self.openprice_long)*Dic.args.nMultiplier*self.nLong      #盈亏金额
            _margin = self.openprice_long*Dic.args.nMultiplier*Dic.args.margin*abs(self.nLong)

            self.balance.append(self.balance[-1] + _margin + _profits- _fees)   #账户余额变动
            # 日志
            Dic.log.csv_trade("{} 平多{} {} {} {:.2f} {:.2f} {:.2f} {:.2f} {:.0f} {:.0f} {:.2f}".format(\
                Dic.code, restr, self.lTimes[-1][0:19], current_time[0:19], self.openprice_long, _open, \
                _open-self.openprice_long, self.nLong, _margin, _profits, _fees
                ))

            self.lTimes.append(current_time)  #交易时间
            self.nLong = 0    #修改当前仓位
            return True

        _open = self.ask_price
        # if _open >=self.openprice_short and self.p_bid!=88 and self.p_ask!=88:
        #     _open = self.openprice_short-Dic.robber_close
        # if _category == 'short' and self.lessthan(self.price_low_1, _open, self.price_heigh_1):
        if _category == 'short':
            # 反向账户
            # if self.a2.nLong!=0  and Dic.isOpenFan==1:
            #     self.a2.close_position("long")

            if _open-self.openprice_short<0:
                self.nsuccess += 1
                self.profitPlus +=1
                self.nClose += 1
                Dic.log.csv_trade("平空收益 {} {:.2f} {} {:.2f} {:.2f} {} {}".format(self.sTimes[-1][0:19], self.openprice_short, current_time[0:19], _open, self.openprice_short-_open, self.nShort, (self.openprice_short-_open)*self.nShort*100))
            else:
                self.nfail += 1
                self.profitMinus += 1
                self.nClose += 1
                Dic.log.csv_trade("平空止损 {} {:.2f} {} {:.2f} {:.2f} {} {}".format(self.sTimes[-1][0:19], self.openprice_short, current_time[0:19], _open, self.openprice_short-_open, self.nShort, (self.openprice_short-_open)*self.nShort*100))

            self.acum = 0  # 记录开仓后至今一共多少K线
            self.sIndexs.append(_open) #

            if "RM" in Dic.code:
                _fees = Dic.args.feeRClose*abs(self.nShort)     #缴纳手续费
            else:
                _fees = _open*Dic.args.nMultiplier*Dic.args.feeRClose*abs(self.nShort)     #缴纳手续费

            _profits = (_open-self.openprice_short)*Dic.args.nMultiplier*self.nShort      #盈亏金额
            _margin = self.openprice_short*Dic.args.nMultiplier*Dic.args.margin*abs(self.nShort)

            self.balance.append(self.balance[-1] + _margin + _profits- _fees)   #账户余额变动
            Dic.log.csv_trade("{} 平空{} {} {} {:.0f} {:.0f} {:.0f} {:.0f} {:.2f} {:.2f} {:.2f}".format(\
                Dic.code, restr, self.sTimes[-1][0:19], current_time[0:19], self.openprice_short, _open, \
                self.openprice_short-_open, _profits, self.nShort, _margin, _profits, _fees
                ))
            self.nShort = 0    #修改当前仓位
            self.sTimes.append(current_time)  #交易时间
            return True
        return False
 
    def trading(self, p_bid, p_ask, current_time, price_close):
        '''
        交易 p_bid: 2为上涨（买多），1为横盘，0为下跌（买空）。
        其他值:3为回测结束，清仓离场
        isSimu True 模拟；isSimu False 实盘
        返回值 2买多， 0买空
        '''
        # if Dic.code == 'SSE.600519':
        #     print("debug")
        self.p_bid = p_bid
        self.p_ask = p_ask
        self.acum +=1

        # nslippage = 0
        # self.current_time = arr[0]
        # self.price_open = arr[1]
        # self.price_heigh = arr[2]
        # self.price_low = arr[3]
        # self.price_close = arr[4]
        self.current_time = current_time
        self.price_close = price_close
        self.bid_price = self.price_close
        self.ask_price = self.price_close

        self.timsspan_longopen += 1
        self.timsspan_shortopen += 1

        # 初始化反向账户
        # self.a2 = a2
        # self.a2.current_time = self.current_time
        # self.a2.price_open = self.price_open
        # self.a2.price_heigh = self.price_heigh
        # self.a2.price_low = self.price_low
        # self.a2.price_close = self.price_close
        # self.a2.bid_price = self.price_close
        # self.a2.ask_price = self.price_close
        # self.a2.p_bid = p_ask
        # self.a2.p_ask = p_bid

        # now_localtime = self.current_time.split(" ")[1].split(".")[0]
        # now_date = self.current_time.split(" ")[0]
        # _now_date = now_date.split("-")[1]+"-"+now_date.split("-")[2]
        # _nShort = self.nShort
        _nLong = self.nLong

        '''
        if p_bid ==88 or p_ask==88 or (now_localtime <= "09:05:00") or ("11：25" < now_localtime <= "13:35") \
            or ("14:55"< now_localtime <= "21:05") or ("22:55:00" < now_localtime ):
            if _nShort != 0:
                self.close_position('short')
            if _nLong != 0:
                self.close_position('long')
            return
        '''
        if p_bid ==88 or p_ask==88:
            # if _nShort != 0:
            #     self.close_position('short')
            if _nLong != 0:
                self.close_position('long')
            return
        #规则1 target=0.05, 平仓：收益5%卖出 or 亏损5%卖出 or 开空：
        # if _nLong != 0:
        #     if (self.bid_price >= self.openprice_long+self.openprice_long*Dic.args.benefit)\
        #         or (self.bid_price <= self.openprice_long-self.openprice_long*Dic.args.benefit) or p_ask in [0]:
        #         self.close_position('long')

        #规则2target=0.05, 收益5%卖出，亏损5%卖出
        # if _nLong != 0:
        #     if (self.bid_price >= self.openprice_long+self.openprice_long*Dic.args.benefit)\
        #         or (self.bid_price <= self.openprice_long-self.openprice_long*Dic.args.benefit):
        #         self.close_position('long')
        
        #规则3  稳定收益 target=0.05, 收益5%卖出，到时间平仓
        if _nLong != 0:
            if (self.bid_price >= self.openprice_long+self.openprice_long*Dic.args.benefit)\
                or self.acum>=Dic.args.closedayes:
                self.close_position('long')

        #规则4 target=0.05, 平仓：收益5%卖出 or 亏损5%卖出 or 开空 稳定收益：
        # if _nLong != 0:
        #     if (self.bid_price >= self.openprice_long+self.openprice_long*Dic.args.benefit)\
        #         or p_ask in [0]:
        #         self.close_position('long')

        #规则5 target=0.05, 平仓：收益5%卖出 or 亏损5%卖出 or 开空 稳定收益：
        # if _nLong != 0:
        #     if (self.bid_price >= self.openprice_long+self.openprice_long*Dic.args.benefit)\
        #         or (self.bid_price <= self.openprice_long-self.openprice_long*Dic.args.benefit) or self.acum>=Dic.args.closedayes:
        #         self.close_position('long')

        #按照信号平仓
        # if p_ask in [2] and _nLong != 0:
        #     self.close_position('long')
        #     self.acum = Dic.tradespantime #-5分钟后再交易
        
        if _nLong ==0 and p_ask in [2]:
            self.longOpen()
    
    def lessthan(self, price1, price, price2):
        if price1<= price <= price2:
            return True
        else:
            return False

