from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR

# def stacker():
def stacker(X_train, X_test, y_train, y_test):

    # create 'estimators' list of tuples:
    estimators = [
        ('ElasticNetReg', ElasticNet(random_state=42)),
        ('LinReg', LinearRegression()),
        ('NNReg', KNeighborsRegressor(n_neighbors=2)),
        ('RandomForestReg', RandomForestRegressor(max_depth=20, random_state=42)),
        ('SVMReg', LinearSVR(random_state=42))
    ]

    # create and train model:
    reg = StackingRegressor(estimators=estimators, 
                            final_estimator = RandomForestRegressor(
                            n_estimators=10, random_state=42))
    reg.fit(X_train, y_train) #.score(X_test, y_test)

    # predicting values:
    y_pred = reg.predict(X_test)

    # My model evaluation: 
    return y_pred

