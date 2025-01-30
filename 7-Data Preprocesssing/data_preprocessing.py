import pandas as pd

df=pd.read_csv('diabetes3.csv')
df
df.describe()

#missing data = > zeros are logically incorrect 
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=0,strategy='mean')
#df.iloc[:,:-1] => dataset Except last column
df.iloc[:,1:-1]=imputer.fit_transform(df.iloc[:,:-1])
df.describe()

#outliers
q1=df.iloc[:,:-1].quantile(.25)
q3=df.iloc[:,:-1].quantile(.75)
IQR=q3-q1
df = df[~((df.iloc[:,:-1] < (q1 - 1.5 * IQR)) |(df.iloc[:,:-1] > (q3 + 1.5 * IQR))).any(axis=1)]
df.describe()
 