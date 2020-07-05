#그레이디언트 소실과 폭주 문제
keras.layers.Dense(10, activation='relu', kernel_initializer='he_normal')

he_avg_init = keras.initializers.VarianceScaling(scale=2., model='fan_avg', distribution='uniform')
keras.layers.Dense(10, activation="sigmoid", kernel_initializer=he_avg_init)

#LeakyRelu
model = keras.models.Sequential([
    keras.layers.Dense(10, kernel_initializer="he_normal"),
    keras.layers.LeakyReLU(alpha=0.2)
])
#selu 를사용
layer =keras.layers.Dense(10, activation="selu", kernel_initializer="lecun_normal")

#배치 정규화 
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300, activation="elu", kernel_initializer ="he_normal")
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="elu", kernel_initializer ="he_normal")
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation="softmax")
])
#첫 번째 배치 정규화 층의 파라미터 살펴보기
[(var.name, var.trainable) for var in model.layers[1].variables]

#활성함수 전에 배치 정규화층을 추가하려면 은닉층에서 활성화 함수를 지정하지 말고 배치 정규화 층 뒤에 별도의 층을 추가해야한다.
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300, activation="elu", kernel_initializer ="he_normal", use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("elu")
    keras.layers.Dense(100, activation="elu", kernel_initializer ="he_normal",  use_bias=False),)
    keras.layers.BatchNormalization(),
    keras.layers.Activation("elu")
    keras.layers.Dense(10, activation="softmax")
])

#그레이디언트 클리핑(역전파될때 일정 임계값을 넘지못하게 자른다)
optimizer = keras.optimizers.SGD(clipvalue=1.0)
model.compile(loss='mse', optimizer=optimizer)

##케라스를 사용한 전이학습
model_A = keras.models.load_model("my_model_A.h5")
model_B_on_A = keras.models.Sequential(model_A.layers[:-1])
model_B_on_A.add(keras.layers.Dense(1, activation="sigmoid"))

model_A_clone = keras.models.clone_model(model_A)
model_A_clone.set.weights(model_A.get_weights())

for layer in model_B_on_A.layers[:=1]:
    layer.trainable = False
model_B_on_A.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])

#재사용된 층의 동결을 해제한 후에 학습률을 낮추는 것이 좋다.
history = model_B_on_A.fit(x_train_B, y_train_B, epochs=4, validation_data=(x_valid_B, y_valid_B))
for layer in model_B_on_A.layers[:=1]:
    layer.trainable = True
optimizer = keras.optimizers.SGD(lr=1e-4)
model_B_on_A.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])
history = model_B_on_A.fit(x_train_B, y_train_B, epochs=16, validation_data=(x_valid_B, y_valid_B))

optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)

#Adam과 Nadam 최적화
optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

#학습률 스케줄링
oprimizer = keras.optimizers.SGD(lr=0.01, decay=1e-4)#decay는 s의 역수입니다.
def exponential_dacay_fn(epoch):
    return 0.01 * 0.1**(epoch / 20)

def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1**(epoch / s)
    return exponential_dacay_fn
exponential_dacay_fn = exponential_decay(lr0=0.01, s =20)

#learning_rate 콜백 함수
lr_scheduler = keras.callbacks.LeaningRateScheduler(exponential_dacay_fn)
history = model.fit(x_train_scaled, y_train,callbacks=[lr_scheduler])

def exponential_decay_fn(epoch, lr):
    return lr * 0.1**(1 / 20)

def piecewise_constant_fn(epoch):
    if epoch < 5:
        return 0.01
    elif epoch < 15:
        return 0.005
    else:
        return 0.001
#최상의 검증 손실이 다섯 번의 연속적인 에포크 동안 향상되지 않을 떄마다 학습률에 0.5*
lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)

s = 20 * len(x_train) //32
learning_rate = keras.optimizers.schedules.ExponentialDecay(0.01, s, 0.1)
optimizer = keras.optimizers.SGD(learning_rate)

##과적합 피하기

#l1, l2 규제
from functools import partial
layer = keras.layers.Dense(100, activation='elu',
                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(0.01))

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28],
    RegularizedDense(300),
    RegularizedDense(100),
    RegularizedDense(10, activation='softmax',
                        kernel_initializer='glorot_uniform')
])

#드롭아웃

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(300, activation='elu', kernel_initializer='he_normal'),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(10, activation='softmax')])

#몬테 카를로 드롭아웃 (훈련된 모델을 재훈련하거나 수정하지않고 성능을 크게향상시킬수있다.)
y_probas = np.stack([model(x_test_scaled, training=True)
                    for sample in range(100)])
y_proba = y_probas.mean(axis=0)


#배치 정규화와 같은 층을 가지고 있다면 dropout층을 다음과 같이 변경
class MCDropout(keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)

#맥스-노름 규제
keras.layers.Dense(100, activation='elu', kernel_initializer="he_normal",
                    kernel_constraint=keras.constraints.max_norm(1.))





