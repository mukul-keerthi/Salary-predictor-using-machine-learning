# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 20:53:39 2020

@author: Mukul
"""

# Polynomial Linear Regression
# In this program we aim to verify if the previous salary claims made by the potential candidate is true or not based on the data collected over similar levels of profile and years of experience
# This comes as a handy tool in determing the reasonable salary for the potential candidate
  
#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import the datasets
dataset = pd.read_csv('Position_Salaries.csv') #Run the entire program if it's causing problem here
#To get a matrix of data
X = dataset.iloc[:, 1:2].values #All data except for the last coloumn; : to get the entire coloumn 1:2 is to get it to matrix form
y = dataset.iloc[:, 2].values #Here 4 is the index of the coloumn

#Splitting dataset into training and test set - Not required
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) #Test size is 20% of the whole dataset, random size used to generate psuedo random number

#Fitting the linear regression into the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#Fitting the polynomial regression into the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4) #Gives the fourth order regression. Increment the order to obtain better fit 
X_poly = poly_reg.fit_transform(X)  #fit_transform method to fit and transform 
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y) #Fit the polynomial regression model

#Visualising the Linear regression results
#First we see the results of linear regression
plt.scatter(X, y, color = 'red') #Provides a scatter plot of the dataset 
plt.plot(X, lin_reg.predict(X), color = 'blue') #Plot of the linear regression of the data
plt.title('Salary Bluff Predictor (Using Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


#Visualising the polynomial regression results
plt.scatter(X, y, color = 'red') #Provides a scatter plot of the dataset 
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue') #lin_reg_2 contains the fit of the polynomial regression model
#Note that we can't just pass (X) as a parameter because X_poly was defined for an existing matrix of features X
plt.title('Salary Bluff Predictor (Using Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Prediction of new result using Linear regression
lin_reg.predict(6.5)  #Here instead of X, we pass as parameter the level of the job

#Prediction of new result using Polynomial regression
lin_reg_2.predict(poly_reg.fit_transform(6.5)) #Here instead of X, we pass as parameter the level of the job


