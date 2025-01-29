import numpy as np
import pandas as pd

data=pd.read_csv('diabetes3.csv')


#min_lim=Q1-1.5*IQR
#max_lim=q3+1.5*IQR

#Let's try to calc it on the first column
first_column=data.iloc[:,[0]].values
q1=np.percentile(first_column, 25)
q3=np.percentile(first_column, 75)
IQR=q3-q1

outlier=[]
for item in first_column:
    if(item<(q1-(1.5*IQR)) or item>(q3+(1.5*IQR))):
        outlier.append(item)
        
    