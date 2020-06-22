import numpy as np
import pandas as pd

datasets = pd.read_csv("./data/csv/iris.csv",
                        index_col=None,
                        header = 0, sep=',')
print(datasets)

print(datasets.head())
print(datasets.tail())

print("================")
print(datasets.values)

aaa = datasets.values
print(type(aaa))

np.save('./data/arr.npy', arr=aaa)

# 넘파이로 저장하시오

