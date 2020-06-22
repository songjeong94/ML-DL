import pandas as pd
import numpy as np

train = pd.read_csv("./data/csv/train-data.csv", index_col=0, header = 0, sep=',')
test = pd.read_csv("./data/csv/test-data.csv", index_col=0, header = 0, sep=',')

print(train)
print(test)
