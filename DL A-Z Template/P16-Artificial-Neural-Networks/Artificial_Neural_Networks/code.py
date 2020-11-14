import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:,3:13].values
y = dataset.iloc[-1].values

from sklearn.preprocessing import LabelEncoder ,OneHotEncoder
label1,label2 = LabelEncoder(),LabelEncoder()
x[:,1]= label1.fit_transform(x[:,1])
x[:,2]= label2.fit_transform(x[:,2])
onehot = OneHotEncoder(categories=np.array(x[:,1]))
z = onehot.fit_transform(x[:,1:2]).toarray()
