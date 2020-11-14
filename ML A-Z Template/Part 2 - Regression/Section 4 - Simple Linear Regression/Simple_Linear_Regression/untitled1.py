# Simple Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1:].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.transform(y_test)

# library take care of feature scaling in simle linear reression 
# we are not gonna do feature scaling


# Fitting SLR to training set
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# getting m and c in y = mx+c
m = regressor.coef_
c = regressor.intercept_

# Predicting the test results
y_pred = regressor.predict(X_test)

# Visualizing the training set results

plt.scatter(X_train, y_train, color = 'red')
# to plot the SLR line we will plot X_pred vs X_train  i.e. predicted value for training set
plt.plot(X_train,regressor.predict(X_train), color='yellow')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Experience(in Years)')
plt.ylabel('Salary (in INR)')

# Visualizing test set results

plt.scatter(X_test, y_pred, color = 'yellow')
plt.scatter(X_test, y_test, color = 'cyan')
plt.show()
print(f"Equation of Line {m} x + {c}")








