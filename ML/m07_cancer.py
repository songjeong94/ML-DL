from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

boston = load_breast_cancer()

x = boston.data
y = boston.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True, random_state = 1)


#model = KNeighborsClassifier()
#model = SVC()
#model = LinearSVC()
#model = KNeighborsRegressor()
#model = RandomForestClassifier()
#model = RandomForestRegressor()


#3.실행

model.fit(x_train, y_train)

#4.평가 예측

y_predict = model.predict(x_test)

score = model.score(x_test, y_test)

# acc = accuracy_score(y_test, y_predict)

print(x_test,"의 예측 결과: ", y_predict )
#print("acc = ", acc)
print("score : ", score)
