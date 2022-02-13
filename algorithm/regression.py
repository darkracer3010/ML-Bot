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
def linear(csv):
    h=[]
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
    X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.33, random_state=42)
    regressor=LinearRegression()
    regressor.fit(X_train,Y_train)
    r2=regressor.score(X_test,Y_test)
    print("Linear Regression Accuracy: "+str(r2*100)+"%")
    plt.title('Linear Regression')
    plt.scatter(X, Y) # actual
    plt.scatter(X, Y_pred, color='red')
    plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red') # predicted
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = 'data:image/png;base64,' + urllib.parse.quote(string)
    html = '<img src = "%s"/>' % uri
    h.append(uri)
    h.append(html)
    return h
def polynomial(csv):
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
    print("Polynomial Regression Accuracy: "+str(r2*100)+"%")
    plt.figure(figsize=(10,5))
    plt.title("Polynomial Regression")
    plt.scatter(X,Y,s=15)
    plt.plot(X,y_poly_pred,color='r')
    fig = plt.gcf()
    buf1 = io.BytesIO()
    fig.savefig(buf1, format='png')
    buf1.seek(0)
    string = base64.b64encode(buf1.read())
    uri = 'data:image/png;base64,' + urllib.parse.quote(string)
    html = '<img src = "%s"/>' % uri
    h.append(uri)
    h.append(html)
    return h
def log(csv):
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
    print("Logistic Regression Accuracy: "+str(score*100)+"%")
    plt.title("Logistic Regression")
    plt.scatter(X, regressor.predict_proba(X)[:,1])
    fig = plt.gcf()
    buf2= io.BytesIO()
    fig.savefig(buf2, format='png')
    buf2.seek(0)
    string = base64.b64encode(buf2.read())
    uri = 'data:image/png;base64,' + urllib.parse.quote(string)
    html = '<img src = "%s"/>' % uri
    h.append(uri)
    h.append(html)
    return h
def dec(csv):
    h=[]
    data=pd.read_csv(csv)
    X=data.iloc[:,:-1].values
    Y=data.iloc[:,-1].values
    X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.33, random_state=42)
    regressor = DecisionTreeRegressor() 
    regressor.fit(X, Y)
    y_pred = regressor.predict(X_test)
    score =r2_score(Y_test, y_pred)
    print("Decision Tree Regression Accuracy: "+str(score*100)+"%")
    X_grid = np.arange(min(X), max(X), 0.01)
    plt.title("Decision Tree Regression")
    X_grid = X_grid.reshape((len(X_grid), 1))  
    plt.scatter(X, Y, color = 'red')
    plt.plot(X_grid, regressor.predict(X_grid), color = 'blue') 
    fig = plt.gcf()
    buf3 = io.BytesIO()
    fig.savefig(buf3, format='png')
    buf3.seek(0)
    string = base64.b64encode(buf3.read())
    uri = 'data:image/png;base64,' + urllib.parse.quote(string)
    html = '<img src = "%s"/>' % uri
    h.append(uri)
    h.append(html)
    return h
def elasti(csv):
    h=[]
    data=pd.read_csv(csv)
    X=data.iloc[:,:-1].values
    Y=data.iloc[:,-1].values
    X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.33, random_state=42)
    enet_model = ElasticNet()
    enet_model.fit(X_train, Y_train)
    y_pred = enet_model.predict(X_test)
    score =r2_score(Y_test, y_pred)
    print("Elasticnet Regression Accuracy: "+str(score*100)+"%")
    plt.title("Elasticnet Regression")
    plt.scatter( X_test, Y_test, color = 'blue' )
    plt.plot( X_test, y_pred, color = 'orange' )
    fig = plt.gcf()
    buf4 = io.BytesIO()
    fig.savefig(buf4, format='png')
    buf4.seek(0)
    string = base64.b64encode(buf4.read())
    uri = 'data:image/png;base64,' + urllib.parse.quote(string)
    html = '<img src = "%s"/>' % uri
    h.append(uri)
    h.append(html)
    return h
def lasso(csv):
    h=[]
    data=pd.read_csv(csv)
    X=data.iloc[:,:-1].values
    Y=data.iloc[:,-1].values
    X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.33, random_state=42)
    lasso=Lasso()
    lasso.fit(X_train, Y_train)
    y_pred=lasso.predict(X_test) 
    score =r2_score(Y_test, y_pred)
    print("Lasso Regression Accuracy: "+str(score*100)+"%")
    plt.title("Lasso Regression")
    plt.scatter( X_test, Y_test, color = 'red' )
    plt.plot( X_test, y_pred, color = 'blue' )
    fig = plt.gcf()
    buf5 = io.BytesIO()
    fig.savefig(buf5, format='png')
    buf5.seek(0)
    string = base64.b64encode(buf5.read())
    uri = 'data:image/png;base64,' + urllib.parse.quote(string)
    html = '<img src = "%s"/>' % uri
    h.append(uri)
    h.append(html)
    return h
def ridge(csv):
    h=[]
    data=pd.read_csv(csv)
    X=data.iloc[:,:-1].values
    Y=data.iloc[:,-1].values
    X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.33, random_state=42)
    rd = Ridge()
    rd.fit(X_train,Y_train)
    y_pred = rd.predict(X_test)   
    score =r2_score(Y_test, y_pred)
    print("Ridge Regression Accuracy: "+str(score*100)+"%")
    plt.title("Ridge Regression")
    plt.scatter( X_test, Y_test, color = 'cyan' )    
    plt.plot( X_test, y_pred, color = 'orange' ) 
    fig = plt.gcf()
    buf6 = io.BytesIO()
    fig.savefig(buf6, format='png')
    buf6.seek(0)
    string = base64.b64encode(buf6.read())
    uri = 'data:image/png;base64,' + urllib.parse.quote(string)
    html = '<img src = "%s"/>' % uri
    h.append(uri)
    h.append(html)
    return h
def check():
    o=linear("data.csv")
    o1=polynomial("data.csv")
    o2=log("data.csv")
    o3=dec("data.csv")
    o4=elasti("data.csv")
    o5=ridge("data.csv")
    o6=lasso("data.csv")
check()