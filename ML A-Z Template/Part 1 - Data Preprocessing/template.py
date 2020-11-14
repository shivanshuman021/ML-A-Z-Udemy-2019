import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values

# run "np.set_printoptions(threshold=np.nan)" if you can't see the full array

# taking care of missing data  
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy = 'mean',axis = 0)

# taking mean of entries of coloumn and replacing the
# missing values with mean (/median/mode-(most_frequent)) fitting imputer object
# to matrix X 

imputer = imputer.fit(X[:,1:3]) #fitting imputer object to matrix X i.e imputer will not work with any other matrix

#above method simply replaces nan with average and returns a new coloumn vector

X[:,1:3] = imputer.transform(X[:,1:3]) # replacing the vector containg nan with the new vector 
X

#Encoding categorical data
## in machine learning we require numerical data so replacing country name with values

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()

# labelencoder_X.fit_transform(X[:,0])  # output =  array([0, 2, 1, 2, 1, 0, 2, 0, 1, 0])

X[:,0] =  labelencoder_X.fit_transform(X[:,0]) 
X


#labelencoder does injustice (in terms of maths) amongst the countries by assigning higher values to one country
#using dummy encoding

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0]) # for coloumn 0 
X = onehotencoder.fit_transform(X).toarray()

# no need to specify which coloumn of X is to be categorized
# since our object operates only on coloumn 1

labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y) 
Y

# splitting the data set into the training and test set

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size =0.2, random_state = 0)

# choosing 20% as test set and rest as training set

# however if we give test_size as .25 in array of ten , dataset will be divided 
#into the trainingset 30% of dataset

# ml model should not learn too much by heart else there would be problem in predicting results

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# don't apply fit_transform to X_test  ; apply transform only because we have to feature
# scale test set on the same scale as  scale in training set

# __ NOTE __ == here we are scaling dummy varaiables as well to get higher accuracy
# in  prediction however by doing this the dummy variable loses physical significance
# thus we may or may not scale dummy variables  ( people have diff opinion on scaling dummy variables)
# scaling dummy variable depends on what case we are talking about






























 