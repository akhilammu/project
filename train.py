import numpy as np
import pandas as pd
data=pd.read_csv("features.csv")
data_new=pd.read_csv("features.csv",na_values=['?'])

data_new.dropna(inplace=True)
predictions=data_new['prediction']
data_new
features_raw = data_new[['Area','Perimeter','Eccentricity']]
from sklearn.model_selection import train_test_split

predict_class = predictions.apply(lambda x: 0 if x == 0 else 1)
np.random.seed(1234)

X_train, X_test, y_train, y_test = train_test_split(features_raw, predict_class, train_size=0.80, random_state=1)


# Show the results of the split
print ("Training set has {} samples." .format(X_train.shape[0]))
print ("Testing set has {} samples." .format(X_test.shape[0]))
import sklearn
from sklearn import svm

C = 1.0
svc = svm.SVC(kernel='linear',C=C,gamma=2)
svc.fit(X_train, y_train)
from sklearn.metrics import fbeta_score
print(X_test)
predictions_test = svc.predict(X_test)
predictions_test

