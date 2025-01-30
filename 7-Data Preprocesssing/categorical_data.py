import pandas as pd
import numpy as np
dataset=pd.read_csv('Data.csv')

x=dataset.iloc[:,0:3].values
y=dataset.iloc[:,3].values

from sklearn.impute import SimpleImputer

imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer = imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
# step 1 -> label encoder => ENcodes labels into 0,1,2,3,..
label_encoder=LabelEncoder()
x[:,0]=label_encoder.fit_transform(x[:,0])
#Not the best


from sklearn.compose import ColumnTransformer
ct=ColumnTransformer([('Country',OneHotEncoder(),[0])],remainder='passthrough')
x=ct.fit_transform(x)
x=x[:,1:]
