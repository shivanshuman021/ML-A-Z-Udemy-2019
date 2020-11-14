# The p-value explains " How likely is to get a result like this "
# if p-value is too small that means that our hypothesis is wrong 
# that's why very small p-values are often ommited

# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, -1] = labelencoder_X.fit_transform(X[:, -1])
onehotencoder = OneHotEncoder(categorical_features = [-1])
X = onehotencoder.fit_transform(X).toarray()


# Avoiding dummy variable trap
X = X[:,1:]

# however linear_model module of scikitlearn library takes care of dummy
# variable so there is no need of removing dummy variable manually here
# however we can do this

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
regressor1 = LinearRegression()
regressor.fit()

# predicting test results
y_pred = regressor.predict(X_test)
y_pred_trained = regressor.predict(X_train)
#Visualiztion can't be done here wrt all variable as we can't plot
#multi dimensional plot on a 2-D page

plt.plot(X_train[:,5],(regressor.predict(X_train))[:,5] , color = 'blue')
plt.scatter(X_test[:,5],y_pred,color = 'red')
plt.xlabel('Marketing Spend')
plt.ylabel('profit')
plt.title('50 Startups in America are making this much profit in effect of marketing')












