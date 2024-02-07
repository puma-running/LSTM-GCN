import pandas as pd
filename ="D:\\pythoncode\\Paper\\regressionDay-PredictNextday-MutilstockGnnbySignal-MyModel\\log\\lr-trade\\trade-20240103151732-加评估指标.xlsx"
df = pd.read_excel(filename,sheet_name = 'trade-20240103151732')
aa = df.loc[df['平仓日期']<'2019-5-1',['收益（含手续费）']]
print(aa)