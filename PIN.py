#-*- coding : utf-8 -*-
# coding: utf-8
"""
@ Calculation of PIN(EKOP Model)
@ author: lbb
@ time: May 23, 2021
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# --------------------------------------------------------------------------------------
# Step 1: Data Processing
# --------------------------------------------------------------------------------------
# Importing data
data = pd.read_csv("sz000001.csv", encoding="unicode_escape")

# Determine whether the order is a buy or sell order
data['Buy Order'] = ''
data['Sell Order'] = ''
BS_data = pd.read_csv("BSData.csv", encoding="unicode_escape")
for i in range(len(data)):
    if i % 10000 == 0:
        print(i)
        print(data.loc[i, 'time'])
    data.loc[i, 'time'] = data.loc[i, 'time'].split(" ")[0] # time format: year/month/day
    now = data.loc[i, 'time']
    if data.loc[i, 'Price'] > ((data.loc[i, 'Buy1'] + data.loc[i, 'Sell1']) / 2.0):
        BS_data.loc[BS_data[(BS_data.Date == now)].index.tolist(), 'B'] += data.loc[i, 'Transactions']
    else:
        BS_data.loc[BS_data[(BS_data.Date == now)].index.tolist(), 'S'] += data.loc[i, 'Transactions']

# Results of data processing: number of daily buy/sell orders
BS_data.to_csv('buysell.csv', index=False, sep=',')


# --------------------------------------------------------------------------------------
# Step 2: Calculate PIN
# --------------------------------------------------------------------------------------
# function getPIN:
# Input: number of daily buy/sell orders in a month
# Output: monthly PIN value
def getPIN(data):
    bs = np.concatenate([data['buy'].to_numpy().reshape(-1, 1), data['sell'].to_numpy().reshape(-1, 1)], axis = 1)
    scale = 1000
    bs = bs / scale
    M = (np.min(bs, axis = 1) + np.max(bs, axis = 1)) / 2
    sell_M = bs[:, 1] - M
    buy_M = bs[:, 0] - M
    sum_M = np.sum(bs, axis = 1) - M

    # Likehood function
    def L(var):
        x = var[2] / (var[2] + var[3])
        tmp1 = np.log(var[0] * (1 - var[1]) * np.exp(-1 * var[3]) * x ** sell_M + var[0] *
                      var[1] * np.exp(-1 * var[3]) * x ** buy_M + (1 - var[0]) * (x ** sum_M))
        tmp2 = np.sum(-2 * var[2] + M * np.log(x) +
                    np.sum(bs, axis=1) * np.log(var[3] + var[2]) + tmp1)
        return -tmp2

    res = minimize(L, x0=np.ones(4) * 0.5, bounds=((0, 1), (0, 1), (0, None), (0, None)))

    # Calculation method of PIN
    def PIN(variable):
        return (variable[0] * variable[3]) / (variable[0] * variable[3] + 2 * variable[2])

    return PIN(variable=res.x)


BS_data['mon'] = BS_data['time'].apply(lambda x:str(x)[:7])
gp = BS_data.groupby('mon')

time = []
pin = []
# Calculate PIN for every month
for name, group in gp:
    p = getPIN(group)
    time.append(name)
    pin.append(p)

# Output the result to file 'PIN.csv'
result = pd.DataFrame({'time':time,'pin':pin})
result.to_csv('PIN.csv',index=None)
