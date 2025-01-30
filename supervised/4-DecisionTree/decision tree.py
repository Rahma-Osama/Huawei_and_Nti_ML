import pandas as pd
data=pd.read_csv('Social_Network_Ads.csv')

x=data.iloc[:,2:4].values
y=data.iloc[:, 4].values

#Normalization
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)
#split data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=.7,random_state=42)

#Training -> Import Algo -> Model
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
# We have 2 functions -> 1-fit=>learn
# 2-predict => test and predict

#train
model.fit(x_train,y_train)
#test
predicted_y=model.predict(x_test)


from sklearn.metrics import accuracy_score,confusion_matrix
cm=confusion_matrix(y_test, predicted_y,)
accuracy = accuracy_score(y_test, predicted_y)
print(accuracy)
#to calc accuracy manual
acc=(cm[0][0]+cm[1][1])/len(y_test) 


