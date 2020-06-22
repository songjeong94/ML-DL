from sklearn.metrics import mean_squared_error
def RSME(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
    print("RSME:", RSME(y_test, y_predict))