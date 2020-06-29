import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="darkgrid") 

# 데이터 불러오기
train_features = pd.read_csv('train_features.csv')
train_target = pd.read_csv('train_target.csv')
test_features = pd.read_csv('test_features.csv')

# 데이터 형태 확인
print(f'train_features {train_features.shape}')
print(f'train_target {train_target.shape}')
print(f'test_features {test_features.shape}')

train_features.head()

train_target.head()

test_features.head()

from sklearn.ensemble import RandomForestRegressor

# id 값을 넣으면 S1 ~ S4 순서대로 시각화
def plot_show(idx, df):
    f, axes = plt.subplots(4, 1)
    f.tight_layout() 
    plt.subplots_adjust(bottom=-0.8)
    for i in range(1, 5):
        axes[i-1].plot(df[df['id']==idx]['S'+str(i)].values)
        axes[i-1].set_title('S'+str(i))
        axes[i-1].set_xlabel('time')

plot_show(1, train_features)
print(train_target[train_target['id']==1])

m_175_df = train_target[train_target['M']==175]['id'].values
m_25_df = train_target[train_target['M']==25]['id'].values

plot_show(m_175_df[1], train_features)

plot_show(m_25_df[1], train_features)

import numpy as np
import matplotlib.pyplot as plt


fs = 5
# sampling frequency 
fmax = 25
# sampling period
dt = 1/fs
# length of signal
N  = 75

df = fmax/N
f = np.arange(0,N)*df

xf = np.fft.fft(train_features[train_features.id==10]['S1'].values)*dt
print(len(np.abs(xf[0:int(N/2+1)])))

plt.plot(f[0:int(N/2+1)],np.abs(xf[0:int(N/2+1)]))
plt.xlabel('frequency(Hz)'); 
plt.ylabel('abs(xf)');
plt.tight_layout()

print(len(np.abs(xf[0:int(N/2)])))
plt.plot(f[0:int(N/2)],np.abs(xf[0:int(N/2)]))
plt.xlabel('frequency(Hz)'); 
plt.ylabel('abs(xf)');
plt.tight_layout()

#푸리에 변환을 하기 전
plt.plot(train_features[train_features.id==10]['S1'].values)

# Fourier transformation and the convolution theorem
def autocorr1(x):
    r2=np.fft.ifft(np.abs(np.fft.fft(x))**2).real
    return r2[:len(x)//2]

plt.plot(autocorr1(train_features[train_features.id==10]['S1'].values))

train_ids = train_features.drop_duplicates(['id'])['id'].values

from tqdm import tqdm

signals = []

for i in tqdm(train_ids):
    xf1 = np.fft.fft(train_features[train_features.id==i]['S1'].values)*dt
    xf2 = np.fft.fft(train_features[train_features.id==i]['S2'].values)*dt
    xf3 = np.fft.fft(train_features[train_features.id==i]['S3'].values)*dt
    xf4 = np.fft.fft(train_features[train_features.id==i]['S4'].values)*dt
    
    signals.append(np.concatenate([np.abs(xf1[0:int(N/2+1)]), np.abs(xf2[0:int(N/2+1)]), np.abs(xf3[0:int(N/2+1)]), np.abs(xf4[0:int(N/2+1)])]))
    
signals = np.array(signals)

model = RandomForestRegressor(
    n_estimators=50,
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features='auto',
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    bootstrap=True,
    oob_score=False,
    n_jobs=-1,
    random_state=0,
    verbose=0,
    warm_start=False,
)

model.fit(signals, train_target)

test_ids = test_features.drop_duplicates(['id'])['id'].values

from tqdm import tqdm

test_signals = []

for i in tqdm(test_ids):
    xf1 = np.fft.fft(test_features[test_features.id==i]['S1'].values)*dt
    xf2 = np.fft.fft(test_features[test_features.id==i]['S2'].values)*dt
    xf3 = np.fft.fft(test_features[test_features.id==i]['S3'].values)*dt
    xf4 = np.fft.fft(test_features[test_features.id==i]['S4'].values)*dt
    
    test_signals.append(np.concatenate([np.abs(xf1[0:int(N/2+1)]), np.abs(xf2[0:int(N/2+1)]), np.abs(xf3[0:int(N/2+1)]), np.abs(xf4[0:int(N/2+1)])]))
    
test_signals = np.array(test_signals)

# 예측 (predict)
y_pred = model.predict(test_signals)
submit = pd.read_csv('sample_submission.csv')

submit.head()

# 답안지에 옮겨 적기
for i in range(4):
    submit.iloc[:,i+1] = y_pred[:,i]
submit.to_csv('Dacon6.csv', index = False)