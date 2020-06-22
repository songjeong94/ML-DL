import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

wine = pd.read_csv("./data/csv/wine_data.csv", index_col = 0 ,header = 0, encoding='cp949', sep=',')
print(wine.shape) 


#최근날짜를 가장 아래로
#wine= wine.sort_values(['178'], ascending=[True])

wine = wine.values

np.save('./data/wine.npy', arr=wine )