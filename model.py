# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

data=pd.read_csv('adding_data.csv')

print(data)

data.info()


print(data.State.unique())


data1=data.drop(['Date_Time'],axis=1)

print(data1)
print(data1.shape)


X=data1.iloc[:,:7]

print(X)


Y=data.iloc[:,-1:]

print(Y)


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators = 100)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(y_pred)

print(X_test)


from sklearn import metrics

print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))


clf.predict([[74.1,36.1,0,0,34.0259715291642,35.4747432769862,102.041146519763]])

print(clf.predict)


import pickle
# open a file, where you ant to store the data
file = open('Boiler_predict.pkl', 'wb')

# dump information to that file

pickle.dump(clf, file)

model = pickle.load(open('Boiler_predict.pkl','rb'))

print(model.predict([[74.1,36.1,0,0,34.0259715291642,35.4747432769862,102.041146519763]]))






