import pandas as pd
dataseet=pd.read_csv('Position_Salaries.csv')


x=dataseet.iloc[:,[1]].values
y=dataseet.iloc[:, 2].values

#consider Dtata is Simple and only for learning so we'll train and predict all data without splitting
#Try Normal Simple Linear Regression

from sklearn.linear_model import LinearRegression
linear_reg1=LinearRegression()

#train
linear_reg1.fit(x,y)
#test
predicted_y=linear_reg1.predict(x)



#visualization for training data 
import matplotlib.pyplot as plt
plt.scatter(x, y,color='green')
predicted_y_train=linear_reg1.predict(x)
plt.plot(x,predicted_y_train,color='red')
plt.title('Training Salery vs Years of experience ##Linear Regression')
plt.xlabel('Years of Experience')
plt.ylabel('Salery')
plt.show()


##########Simple Linear Regression is not effeciant as line is far from other data points

###### let's Try Polynomial Regression
##Same as linear but we change inpt style to form another equation from different degree
## for ex degree =2 ax^2+bx+c
## we try different degrees until we reach the nearest 
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=2)
x_poly=poly.fit_transform(x)
linear_reg2=LinearRegression()
# We have 2 functions -> 1-fit=>learn
# 2-predict => test and predict

#train
linear_reg2.fit(x_poly,y)
#test
predicted_y2=linear_reg2.predict(x_poly)

#visualization for training data 
import matplotlib.pyplot as plt
plt.scatter(x, y,color='green')
plt.plot(x,predicted_y2,color='red')
plt.title('Training Salery vs Years of experience ##Polynomial Regression Degree 2')
plt.xlabel('Years of Experience')
plt.ylabel('Salery')
plt.show()

## Repeat but with higher degree ==> More accurate
poly3=PolynomialFeatures(degree=3)
x_poly3=poly3.fit_transform(x)
linear_reg3=LinearRegression()
# We have 2 functions -> 1-fit=>learn
# 2-predict => test and predict

#train
linear_reg3.fit(x_poly3,y)
#test
predicted_y3=linear_reg3.predict(x_poly3)

#visualization for training data 
import matplotlib.pyplot as plt
plt.scatter(x, y,color='green')
plt.plot(x,predicted_y3,color='red')
plt.title('Training Salery vs Years of experience ##Polynomial Regression Degree 3')
plt.xlabel('Years of Experience')
plt.ylabel('Salery')
plt.show()

## Repeat but with higher degree ==> More accurate
poly4=PolynomialFeatures(degree=4)
x_poly4=poly4.fit_transform(x)
linear_reg4=LinearRegression()
# We have 2 functions -> 1-fit=>learn
# 2-predict => test and predict

#train
linear_reg4.fit(x_poly4,y)
#test
predicted_y4=linear_reg4.predict(x_poly4)

#visualization for training data 
import matplotlib.pyplot as plt
plt.scatter(x, y,color='green')
plt.plot(x,predicted_y4,color='red')
plt.title('Training Salery vs Years of experience ##Polynomial Regression Degree 4')
plt.xlabel('Years of Experience')
plt.ylabel('Salery')
plt.show()

##The Best Result is for degree = 4
