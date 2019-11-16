#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 15:59:33 2019

@author: Polina Bondarenko
"""

import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression

import pandas as pd

import pandas_datareader.data as web

import matplotlib.pyplot as plt

'''
predictLinear takes in a ticker, a start date, and the number of days in the future.
It computes a prediction of the stock price in the future using Linear Regression.
'''

def predictLinear(ticker, start_date, days_in_future):
    
    end = datetime.now()
    
    # Retrieves stock data using Pandas DataReader
    df = web.DataReader(ticker, "yahoo", start_date, end)
    
    df.to_csv(ticker + "_history.csv")
    
    # Retrieve close values of the stock for every single day
    close_vals = df['Close'].values
   
    # Make a list of numbers that correspond to a date, i.e. 0 -> 1/1/2017, 1 -> 1/2/2017
    dates = np.arange(len(df))
    
    plt.plot(dates, close_vals)
    
    # Generate matrix to feed into Linear Regression algorithm
    Mat = np.zeros((len(dates), 2))
    
    # First column is a vector of ones
    Mat[:, 0] = np.ones(len(dates))
    # Second column column is our data (x-values)
    Mat[:, 1] = dates
    
    # Genderate Linear Regression model
    model = LinearRegression().fit(Mat, close_vals)
    coeffs = model.coef_
    intercept = model.intercept_
    
    # Graphing
    a = np.linspace(0, len(Mat), 10000)
    b = model.intercept_ + coeffs[1]*a
    
    plt.plot(dates, close_vals, color='b')
    plt.plot(a, b, color='r')
    plt.title("Linear Regression Model for " + ticker + " starting at " + start_date.strftime('%m-%d-%Y'))
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    
    # Compute prediction using computed coefficients
    # y = b + ax
    # x is the number od days in the future plus number of dates we have used
    # b is the intercept
    # a is the coeffs[1]
    # y is the prediction
    prediction = intercept + coeffs[1] * (len(dates) + days_in_future - 1)
    
    return prediction

tickers = input("Enter a list of tickers separated by commas: ")
ticker_array = tickers.split(', ')
start_date = input("Enter a date (MM-DD-YYYY): ")
start_date = datetime.strptime(start_date, '%m-%d-%Y')
days_in_future = int(input("Enter the number of days in the future: "))

for ticker in ticker_array:
    prediction = predictLinear(ticker, start_date, days_in_future)
    print(ticker + " price in " + str(days_in_future) + " days will be $" 
              + str(round(prediction, 2)) + " according to this model")
    predictLinear(ticker, start_date, days_in_future)
    plt.show()
    print("****************************************************************")



















