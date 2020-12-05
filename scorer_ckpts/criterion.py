import os 
import pandas as pd
import numpy as np
for task in os.listdir('.'):
    if task.find('.py') != -1:
        continue
    data = pd.read_csv(open(task + '/criterion.csv', 'r', encoding='utf-8'))
    label_mean = sum(data['label']) / len(data['label'])
    output_mean = sum(data['property']) / len(data['property'])
    ssreg = sum([(num - label_mean)**2 for num in data['property']])
    sstot = sum([(num - label_mean)**2 for num in data['label']])
    
    mse = 0.0
    mae = 0.0
    for i in range(len(data['label'])):
        mse += (data['label'][i] - data['property'][i])**2
        mae += abs(data['label'][i] - data['property'][i])
    mse = mse / len(data['label'])
    mae = mae / len(data['label'])
    rmse = mse**(1/2)
    print("{}:\n\tR2: {:.4f}\trmse: {:.4f}\tmse: {:.4f}\tmae: {:.4f}".format(task, ssreg/sstot, rmse, mse, mae))