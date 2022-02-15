from operator import mod
from tkinter.messagebox import NO
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.metrics import accuracy_score,r2_score
from sklearn.linear_model import LogisticRegression,ElasticNet,LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
import os
import sys
import matplotlib
import io
import matplotlib.pyplot as plt
from io import StringIO
import urllib, base64
model=None
dm={'Linear':None,'Polynomial':None,'Logistic':None,'Decision Tree':None,'Lasso':None,'Ridge':None,'ElasticNet':None}
def linear(csv):
    global dm
    h=[]
    o=[]
    data=pd.read_csv(csv)
    X=data.iloc[:,:-1]
    Y=data.iloc[:,-1]
    X_mean=np.mean(X)
    Y_mean=np.mean(Y)
    num=0
    den=0
    for i in range(len(X)):
        num+=(X.iloc[i] - X_mean)*(Y.iloc[i] - Y_mean)
        den+=(X.iloc[i] - X_mean)**2
    m = num / den
    c = Y_mean - m*X_mean
    Y_pred = m*X + c
    o.append(m)
    o.append(c)
    X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.33, random_state=42)
    regressor=LinearRegression()
    regressor.fit(X_train,Y_train)
    r2=regressor.score(X_test,Y_test)
    h.append(r2)
    dm['Linear']=o
    return h
def polynomial(csv):
    global dm
    h=[]
    data=pd.read_csv(csv)
    X=data.iloc[:,:-1].values
    Y=data.iloc[:,-1].values
    polynomial_features=PolynomialFeatures(degree=2)
    x_poly=polynomial_features.fit_transform(X)
    regressor=LinearRegression()
    regressor.fit(x_poly, Y)
    y_poly_pred=regressor.predict(x_poly)
    r2=r2_score(Y,y_poly_pred)
    h.append(r2)
    dm['Polynomial']=regressor
    return h
def log(csv):
    global dm
    h=[]
    data=pd.read_csv(csv)
    X=data.iloc[:,:-1].values
    Y=data.iloc[:,-1].values
    X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.33, random_state=42)
    sc_x = StandardScaler()
    X_train=sc_x.fit_transform(X_train) 
    X_test=sc_x.transform(X_test)
    regressor=LogisticRegression()
    regressor.fit(X_train, Y_train)
    y_pred = regressor.predict(X_test)
    score =r2_score(Y_test, y_pred)
    dm['Logistic']=regressor
    h.append(score)
    return h
def dec(csv):
    global dm
    h=[]
    data=pd.read_csv(csv)
    X=data.iloc[:,:-1].values
    Y=data.iloc[:,-1].values
    X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.33, random_state=42)
    regressor = DecisionTreeRegressor() 
    regressor.fit(X, Y)
    y_pred = regressor.predict(X_test)
    score =r2_score(Y_test, y_pred)
    dm['Decision Tree']=regressor
    h.append(score)
    return h
def elasti(csv):
    global dm
    h=[]
    data=pd.read_csv(csv)
    X=data.iloc[:,:-1].values
    Y=data.iloc[:,-1].values
    X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.33, random_state=42)
    enet_model = ElasticNet()
    enet_model.fit(X_train, Y_train)
    y_pred = enet_model.predict(X_test)
    score =r2_score(Y_test, y_pred)
    dm['ElasticNet']=enet_model
    h.append(score)
    return h
def lasso(csv):
    global dm
    h=[]
    data=pd.read_csv(csv)
    X=data.iloc[:,:-1].values
    Y=data.iloc[:,-1].values
    X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.33, random_state=42)
    lasso=Lasso()
    lasso.fit(X_train, Y_train)
    y_pred=lasso.predict(X_test) 
    score =r2_score(Y_test, y_pred)
    dm['Lasso']=lasso
    h.append(score)
    return h
def ridge(csv):
    global dm
    h=[]
    data=pd.read_csv(csv)
    X=data.iloc[:,:-1].values
    Y=data.iloc[:,-1].values
    X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.33, random_state=42)
    rd = Ridge()
    rd.fit(X_train,Y_train)
    y_pred = rd.predict(X_test)   
    score =r2_score(Y_test, y_pred)
    dm['Ridge']=rd
    h.append(score)
    return h
def check(data):
    o=linear(data)
    o1=polynomial(data)
    o2=log(data)
    o3=dec(data)
    o4=elasti(data)
    o5=ridge(data)
    o6=lasso(data)
    l1=["Linear","Polynomial","Logistic","Decision Tree","ElasticNet","Ridge","Lasso"]
    p=[o,o1,o2,o3,o4,o5,o6]
    ind=p.index(max(p))
    global model
    model=l1[ind]
check("data.csv")
def predict(X):
    global model,dm
    if(model!="linear"):
        return float(dm[model].predict(X))
    return float(dm['Linear'][0]*X +dm['Linear'][1])
print(predict(np.array([3000]).reshape(-1,1)))
