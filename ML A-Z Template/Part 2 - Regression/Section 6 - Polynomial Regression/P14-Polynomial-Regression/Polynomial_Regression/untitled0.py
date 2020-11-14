# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
# X = dataset.iloc[:, 1].values  # it will give X as a vector but it should be a matrix
X = dataset.iloc[:, 1:2].values # X is a matrix 
y = dataset.iloc[:, -1].values  # y is a vector

# Here data set is too small we will not split it into training and test set
"""
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""
# Linear regression will do feature scalin automatically
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

# Fitting Polynomial Regression to the Dataset

from sklearn.preprocessing import PolynomialFeatures
poly_reg2 = PolynomialFeatures(degree=2)
#poly_reg object of PolynomialFeatures class is a tool that transforms our matrix of features 
# X into  a matrix of polynomial features i.e. X squared , X cubed etc..
X_poly2= poly_reg2.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly2,y)



# Visualization of the Linear regression result
plt.scatter(X,y,color ='violet')
plt.plot(X,lin_reg.predict(X),color = 'red')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
# Visualization of the Polynomial regression result
plt.scatter(X,y,color ='violet')
plt.plot(X,lin_reg2.predict(poly_reg2.fit_transform(X)),color = 'red')
# poly_reg.fit_transform(X) in bracket because using this ; if we add some more features to our matrix 
# this will give results
# However if we use X_poly which is already defined it will not be an easy task 
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


## Degree = 3 graph more accurte

from sklearn.preprocessing import PolynomialFeatures
poly_reg3 = PolynomialFeatures(degree=3)
X_poly3= poly_reg3.fit_transform(X)
lin_reg3 = LinearRegression()
lin_reg3.fit(X_poly3,y)



plt.scatter(X,y,color ='violet')
plt.plot(X,lin_reg3.predict(poly_reg3.fit_transform(X)),color = 'red')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


## Degree 4 graph
from sklearn.preprocessing import PolynomialFeatures
poly_reg4 = PolynomialFeatures(degree=4)
X_poly4= poly_reg4.fit_transform(X)
lin_reg4 = LinearRegression()
lin_reg4.fit(X_poly4,y)



plt.scatter(X,y,color ='violet')
plt.plot(X,lin_reg4.predict(poly_reg4.fit_transform(X)),color = 'red')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()



## Degree 4 graph
# To make curve more smooth
from sklearn.preprocessing import PolynomialFeatures
poly_reg4 = PolynomialFeatures(degree=4)
X_poly4= poly_reg4.fit_transform(X)
lin_reg4 = LinearRegression()
lin_reg4.fit(X_poly4,y)


X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color ='violet')
plt.plot(X_grid,lin_reg4.predict(poly_reg4.fit_transform(X_grid)),color = 'red')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# Predicting a new result with Linear Regression
lin_pred = lin_reg.predict(X) # predicts salary corresponding to level 6.5

# Predicting a new result Poly_reg4
poly_pred4 = lin_reg4.predict(poly_reg4.fit_transform(X))

