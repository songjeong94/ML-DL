import numpy as np

def outliers(data):
    for i in data:
        quartile_1, quartile_3 = np.percentile(data, [25, 75])
        data = np[:, i : i+1]
        print("1사분위: ", quartile_1)
        print("3사분위: ", quartile_3)
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        print(np.where((data>upper_bound) | (data<lower_bound)))

data = np.array([1,2,3,100,5,6,7], [3,4,5,6,100,200,9])
# 실습 : 행렬을 입력해서 컬럼별로 이상치 발견하는 함수를 구현하시오
# 파일명 : m36_outlier2.py



def outliers(train):
    for i in train:
        train_out = train[:,i:i+1]
        for i in train:
            train_out = train[:,i:i+1]
        quartile_1, quartile_3 = np.percentile(train_out, [25, 75])
        print("1사분위: ", quartile_1)
        print("3사분위: ", quartile_3)
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        print(np.where((train_out>upper_bound) | (train_out<lower_bound)))

b = outliers(train)
