# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the Dataset
dataset = pd.read_csv("Covid_cases_in _Karnataka.csv")

#Filtering the needed coloumns
dataset1=dataset.filter(['Date','Total cases'])

#Adding the Duplicate Coloumns
dataset1 = dataset1.groupby(['Date']).sum()

#Reseting the dataset1
dataset1=dataset1.reset_index()

#Spliting the Date Coloumn
dataset1['Date'] = pd.to_datetime(dataset1['Date'])
dataset1.insert(1, "year", dataset1.Date.dt.year, True)
dataset1.insert(2, "month", dataset1.Date.dt.month, True)
dataset1.insert(3, "Day", dataset1.Date.dt.day, True)

#Droping the Date coloumn
dataset1.drop('Date', axis=1, inplace=True)

#Dividing the dataset1 into X and Y Matrix
X=dataset1.filter(['year','month','Day'])
Y=dataset1.filter(['Total cases']) 

#Dividing the dataset1 into training and testing data
#from sklearn.model_selection import train_test_split
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 0)

#Training the Model with train data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

#Predecting the Model by giving Values
Y_pred=lin_reg_2.predict(poly_reg.fit_transform([['2020','3','10']]))


