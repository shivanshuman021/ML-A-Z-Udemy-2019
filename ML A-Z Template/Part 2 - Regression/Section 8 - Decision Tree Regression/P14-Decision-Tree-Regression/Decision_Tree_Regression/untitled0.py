import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Dataset = pd.read_csv('Position_Salaries.csv')

X = Dataset.iloc[:,1:2].values
y = Dataset.iloc[:,2].values 
# NO FEATURE SCALING
# NO SPLITTING OF DATASET

# FITTING REGRESSION MODEL
from sklearn.tree import DecisionTreeRegressor
tregressor = DecisionTreeRegressor(random_state = 0)
tregressor.fit(X,y)

a = np.array([6.5])
a = a.reshape(1,1)

y_pred = tregressor.predict(a)

# in decision tree the prediction in an interval is the average value in the interval
# But plotting graph by conventional plotting code is giving staraight lines 
# in each interval
# so this code is not applicable
plt.scatter(X,y,color = 'red')
plt.plot(X,tregressor.predict(X))
plt.show()

# Here if we apply a code that plots higher resolution graphs SET RESOLUTION PARAMETER 
#AS PER NEED HERE IT IS 0.01
X_grid = np.arange(min(X),max(X),0.0001)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,tregressor.predict(X_grid),color = 'green')
plt.show()



#### CONCLUSION - DECISION TREE IS A NON CONTINUOUS MACHINE LEARNING MODEL