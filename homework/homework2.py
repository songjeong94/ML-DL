#회귀용 다층 퍼셉트론 만들기 p383
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()

x_train_full, x_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target)
x_train, x_valid,y_train,y_valid = train_test_split(
    x_train_full, y_train_full)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_valid = scaler.fit_transform(x_valid)
x_test = scaler.fit_transform(x_test)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=x_train_shape[1:]),
    keras.layers.Dense(1)
])
model.compile(loss="mean_squared_error", optimizer='sgd')
history = model.fit(x_train,y_train,epochs=20, validation_data(x_valid, y_valid))
mse_test = model.evaluate(x_test,y_test)
x_new = x_test[:3]
y_pred = model.predict(x_new)

#함수형 api를 사용해 복잡한 모델 만들기 p385
input_ = keras.layers.Input(shape=x_train.shape[1:])
hidden1 = keras.layers.Dense(30, activation="relu")(input_)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.Concatenate()([input_, hidden2])
output = keras.layers.Dense(1)(concat)
model = keras.Model(inputs=[input_], outputs=[output])

#입력 2개 받기 p386
input_A = keras.layers.Input(shape=[5], name='wide_input')
input_B = keras.layers.Input(shape=[6], name='deep_input')
hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.Concatenate()([input_A, hidden2])
output = keras.layers.Dense(1, name='output')(concat)
model = keras.Model(inputs=[input_A, input_B], outputs=[output])

model.compile(loss='mse', optimizer=keras.optimizers.SGD(lr=1e-3))

x_train_A, x_trian_B = x_train[:,:5],x_train[:,2:]
x_valid_A, x_valid_B = x_valid[:,:5],x_valid[:,2:]
x_test_A, x_test_B = x_test[:,:5],x_test[:,2:]
x_new_A, x_new_B = x_test_A[:,:5],x_test_B[:,2:]

history = model.fit((x_train_A,x_trian_B),y_train,epochs=20, validation_data((x_valid_A,x_valid_B), y_valid))
mse_test = model.evaluate((x_test_A,x_test_B),y_test)

#은닉층에서의 보조출력
ouput = keras.layers.Dense(1, name="main_output")(concat)
aux_output = keras.layers.Dense(1, name="aux_output")(hidden2)
model = keras.Model(inputs=[input_A,input_B], outputs=[output, aux_output])
model.compile(loss=["mse","mse"], loss_weights=[0.9,0.1], optimizer='sgd')

history = model.fit([x_train_A,x_trian_B],[y_train,y_train],epochs=20, validation_data([x_valid_A,x_valid_B], [y_valid, y_valid))
total_loss, main_loss, aux_loss = model.evaluate([x_test_A,x_test_B],[y_test,y_test])
y_pred_main, y_pred_aux = model.predict([x_new_A,x_new_B])

#389 서브 클래싱 api로 동적 모델 만들기
class WideAndDeepModel(keras.Model):
    def __init__(self, units=30, activation='relu', **kwargs):
        super().__init__(**kwargs) #표준 매개변수 처리
        self.hidden1 = keras.layers.Dense(units, activation=activation)
        self.hidden2 = keras.layers.Dense(units, activation=activation)
        self.main_output = keras.layers.Dense(1)
        self.aux_output = keras.layers.Dense(1)
    
    def call(self, inputs):
        input_A, input_B = inputs
        hidden1 = self.hidden1(input_B)
        hidden2 = self.hidden2(hidden1)
        concat = keras.layers.concatenate([input_A, hidden2])
        main_output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return main_output, aux_output

model = WideAndDeepModel()

#모델 저장과 복원
model.save("my_keras_model.h5")
model = keras.models.load_model("my_keras_model.h5")

#콜백 사용하기
check_point_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5")
history = model.fit(x_train, y_train, epochs=10, callbacks=[checkpount_cb])

#텐서보드
import os
root_logdir = os.path.join(os.curdir, "my_logs")

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()

trnsorboard_cb = keras.callbacks.TensorBoard(run_logdir)
histort = model.fit(x_train,y_train,epochs=30,validation_data=(x_valid,y_valid),callbacks=[tensorboard_cb])

#신경망 하이퍼파라미터 튜닝하기
def build_model(n_hidden=1, n_neurnos=30, learning_rate=3e-3, input_shape=[8]):
    model =keras.model.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation='relu'))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(lr=learning_rate)
    model.compile(loss='mse', optimizer=optimizer)
    return model

keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model) #모델 래퍼를통해 객체만들기

keras_reg.fit(x_train, y_train, epochs=100, validation_data=(x_valid, y_valid),
                callbacks=[keras.callbacks.EarlyStopping(patience=10)])
mse_test = keras_reg.score(x_test, y_test)
y_pred = keras_reg.predict(x_new)

from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

param_distribs = {
    "n_hidden": [0, 1, 2, 3],
    "n_neurons": np.arange(1, 100),
    "learning_rate": reciprocal(3e-4, 3e-2)
}

rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3)
rnd_search_cv.fit(x_train, y_train, epochs=100,
                    validation_data=(x_valid, y_valid),
                    callbacks = [keras.callbacks.EarlyStopping(patience=10)]
)




