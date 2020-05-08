from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR



# def LinReg():
def LinReg(X_train, X_test, y_train, y_test):
    # creating and training
    lm = LinearRegression()
    lm.fit(X_train,y_train)

    # predicting values:
    y_pred = lm.predict(X_test)

    # My model evaluation: 
    return y_pred


# def ElasticNetReg():
def ENReg(X_train, X_test, y_train, y_test):
    # creating and training
    en = ElasticNet(random_state=42)
    en.fit(X_train,y_train)

    # predicting values:
    y_pred = en.predict(X_test)

    # My model evaluation: 
    return y_pred


# def NNReg():
def NNReg(X_train, X_test, y_train, y_test):

    # creating and training
    NNReg = KNeighborsRegressor(n_neighbors=2)
    NNReg.fit(X_train,y_train)

    # predicting values:
    y_pred = NNReg.predict(X_test)

    # My model evaluation: 
    return y_pred


# def RandomForestReg():
def RandomForestReg(X_train, X_test, y_train, y_test):

    # train model:
    regr = RandomForestRegressor(max_depth=20, random_state=42) 
    regr.fit(X_train, y_train)

    # predict values:
    y_pred = regr.predict(X_test)

    # validate
    return y_pred


# def SVMReg():
def SVMReg(X_train, X_test, y_train, y_test):

    # creating and training
    SVMReg = LinearSVR(random_state=42)
    SVMReg.fit(X_train,y_train)

    # predicting values:
    y_pred = SVMReg.predict(X_test)

    # My model evaluation: 
    return y_pred