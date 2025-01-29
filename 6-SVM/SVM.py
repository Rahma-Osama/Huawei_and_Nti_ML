import pandas as pd
data=pd.read_csv('Social_Network_Ads.csv')

x=data.iloc[:,2:4].values
y=data.iloc[:, 4].values

#Normalization
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit_transform(x)
#split data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=.7,random_state=42)

#Training -> Import Algo -> Model
from sklearn.svm import SVC
#from sklearn.svm import SVR --> Regression

#We need to run the 4 types of SVM 
linearSVC=SVC(kernel='linear')
# We have 2 functions -> 1-fit=>learn
# 2-predict => test and predict
#train
linearSVC.fit(x_train,y_train)
#test
predicted_y=linearSVC.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predicted_y)
print("Linear SVC : ",accuracy)



sigmoidSVC=SVC(kernel='sigmoid')
sigmoidSVC.fit(x_train,y_train)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, sigmoidSVC.predict(x_test))
print("sigmoid SVC : ",accuracy)


polyrSVC=SVC(kernel='poly')
polyrSVC.fit(x_train,y_train)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, polyrSVC.predict(x_test))
print("Poly SVC : ",accuracy)


rbfSVC=SVC(kernel='rbf')
rbfSVC.fit(x_train,y_train)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, rbfSVC.predict(x_test))
print("Rbf SVC : ",accuracy)
