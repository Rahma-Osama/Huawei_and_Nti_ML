import pandas as pd
dataseet=pd.read_csv('Salary_Data.csv')


x=dataseet.iloc[:,[0]].values
y=dataseet.iloc[:, 1].values

#split data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=.7,random_state=42)

#Training -> Import Algo -> Model
from sklearn.linear_model import LinearRegression
model=LinearRegression()
# We have 2 functions -> 1-fit=>learn
# 2-predict => test and predict

#train
model.fit(x_train,y_train)
#test
predicted_y=model.predict(x_test)



#visualization for training data 
import matplotlib.pyplot as plt
plt.scatter(x_train, y_train,color='green')
predicted_y_train=model.predict(x_train)
plt.plot(x_train,predicted_y_train,color='red')
plt.title('Training Salery vs Years of experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salery')
plt.show()

#visualization for testingv data 
plt.scatter(x_test, y_test,color='green')
plt.plot(x_test,predicted_y,color='red')
plt.title('Training Salery vs Years of experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salery')
plt.show()


#Y=wX+b
w=model.coef_
y=model.intercept_
print("")


#Model Evaluation
from sklearn.metrics import mean_absolute_error , mean_squared_error,r2_score
#accuracy --> classification
#MAE , MSE , R square
mae=mean_absolute_error(y_test, predicted_y)
print("MAE : ",mae)
mse=mean_squared_error(y_test,predicted_y)
print("MSE : ",mse)
r2=r2_score(y_test,predicted_y)  #the nearset to 1 the more accurate
print("R2 : ",r2)


print(model.predict([[6.8]]))