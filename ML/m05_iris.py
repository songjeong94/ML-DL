from sklearn.datasets import load_iris
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

iris = load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True, random_state = 1)


#model = KNeighborsClassifier(n_neighbors=1)
#model = SVC()
#model = LinearSVC()
#model = KNeighborsRegressor()
#model = RandomForestClassifier()
model = RandomForestRegressor()

#3.실행

model.fit(x_train, y_train)

#4.평가 예측

y_predict = model.predict(x_test)
score = model.score(x_test, y_test)

#acc = accuracy_score(y_test, y_predict)

print(x_test,"의 예측 결과: ", y_predict )
#print("acc = ", acc)
print("score:", score) #분류일떄는 acc 회귀일때는 r2
