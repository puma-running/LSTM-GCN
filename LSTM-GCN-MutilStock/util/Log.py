# coding=utf-8
import logging
import time
import os
import csv
  
class Log(object):
    def __init__(self, filename):
        log_path = './log'
        if os.path.exists(log_path) and os.path.isdir(log_path):
            pass
        else:
            os.mkdir(log_path)
        self.logname = os.path.join(log_path, '{0}'.format(filename))
        # Create a logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        # Create a handler for writing log files
        fh = logging.FileHandler(self.logname, 'a', encoding='utf-8')
        # fh.setLevel(logging.DEBUG)
        fh.setLevel(logging.INFO)
        # Create a handler for output to the console
        ch = logging.StreamHandler()
        # ch.setLevel(logging.DEBUG)
        ch.setLevel(logging.INFO)
        # Define the output format of handler
        #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter('%(asctime)s  - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # Add a handler to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)
        self.logger = logger
        self.__fh = fh
        self.__ch = ch
        _now = time.strftime("%Y%m%d%H%M%S")

        filename1 = os.path.join(log_path, 'trade-{0}.csv'.format(_now))
        csv_trade = open(filename1, 'w+', newline="")
        self.writer_trade = csv.writer(csv_trade)
        self.writer_trade.writerow(["stockCode","ProfitandLoss","openDate","openTime","closeDate","closeTime","openPrice","closePrice","DifferenceinPrice","quantity","funds","Revenue (including fees)","fees"])

        filename2 = os.path.join(log_path, 'performance-metrics-{0}.csv'.format(_now))
        csv_performance = open(filename2, 'w+', newline="")
        self.writer_performance = csv.writer(csv_performance)
        self.writer_performance.writerow(["code","mape","mdape","acc","pre_pos","pre_neg","rec_pos","rec_neg","F1_pos","F1_neg"])
    
        filename2 = os.path.join(log_path, 'performance-values-{0}.csv'.format(_now))
        csv_performance = open(filename2, 'w+', newline="")
        self.writer_performance_values = csv.writer(csv_performance)
        self.writer_performance_values.writerow(["code","txt_inputs","txt_outputs","txt_targets"])
    def closeLog(self):
        # Remove the Handler after logging
        self.logger.removeHandler(self.__fh)
        self.logger.removeHandler(self.__ch)
 
        # Close the open file
        self.__fh.close()
        self.__ch.close()
 
    def __printconsole(self, level, message):
        # Record a journal
        if level == 'info':
            self.logger.info(message)
        elif level == 'debug':
            self.logger.debug(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)
 
    def debug(self, message):
        self.__printconsole('debug', message)
 
    def info(self, message):
        self.__printconsole('info', message)

    def warning(self, message):
        self.__printconsole('warning', message)
    def error(self, message):
        self.__printconsole('error', message)

    def csv_trade(self, message):
        self.writer_trade.writerow(message.split(" "))

    def csv_performance_metrics(self, message):
        self.writer_performance.writerow(message.split(" "))

    def csv_performance_values(self, str, txt_inputs, txt_outputs, txt_targets):
        self.writer_performance_values.writerow([str, txt_inputs, txt_outputs, txt_targets])