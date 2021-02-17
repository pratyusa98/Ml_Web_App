import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor,LGBMClassifier
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score

#regression

def pred_regression(X, y,cols):
    # data read
    df = pd.read_csv(X)
    df.drop(cols, axis=1, inplace=True)
    df.fillna(0,inplace=True)

    # label enc
    df = pd.get_dummies(data=df, drop_first=True)

    # split x and y
    X = df.drop(y, axis=1)
    y = df[y]

    # feature selection
    def constant(X, thres):
        var_thres = VarianceThreshold(threshold=thres)
        var_thres.fit(X)
        constant_columns = [column for column in X.columns
                            if column not in X.columns[var_thres.get_support()]]
        return constant_columns

    constant_columns = constant(X, 1)
    X.drop(constant_columns, axis=1, inplace=True)

    # train_test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=45)

    # model Build
    lgbr = LGBMRegressor().fit(X_train, y_train)
    rfr = RandomForestRegressor().fit(X_train, y_train)

    # result
    y_pred = lgbr.predict(X_test)
    y_pred_rfr = rfr.predict(X_test)

    #LGBM
    rmse = mean_squared_error(y_test, y_pred, squared=False).round(3)
    score = r2_score(y_test, y_pred).round(3)

    #RFR
    rmse_rfr = mean_squared_error(y_test, y_pred_rfr, squared=False).round(3)
    score_rfr = r2_score(y_test, y_pred_rfr).round(3)

    return score,rmse,rmse_rfr,score_rfr



#classification
def pred_classification(X, y,cols):
    # data read
    df = pd.read_csv(X)

    df.drop(cols, axis=1, inplace=True)
    df.fillna(0, inplace=True)

    # label enc
    df = pd.get_dummies(data=df, drop_first=True)

    colss = len(df.columns)


    # split x and y
    X = df.drop(y, axis=1)
    y = df[y]

    # feature selection
    def constant(X, thres):
        var_thres = VarianceThreshold(threshold=thres)
        var_thres.fit(X)
        constant_columns = [column for column in X.columns
                            if column not in X.columns[var_thres.get_support()]]
        return constant_columns

    constant_columns = constant(X, 1)
    X.drop(constant_columns, axis=1, inplace=True)

    # train_test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=45)

    # model Build
    lgbc = LGBMClassifier().fit(X_train, y_train)
    rfc = RandomForestClassifier().fit(X_train, y_train)

    # result
    y_pred = lgbc.predict(X_test)
    y_pred_rfc = rfc.predict(X_test)

    f1score = accuracy_score(y_test, y_pred).round(3)
    f1score_rfc = accuracy_score(y_test, y_pred_rfc).round(3)
    return f1score,f1score_rfc