import pandas as pd
import numpy as np

dataset=pd.read_csv('Wuzzuf_Jobs.csv')

""" 
Wuzzuf jobs in Egypt data set at Kaggle
https://www.kaggle.com/omarhanyy/wuzzuf-jobs
Build all python needed classes to get the following from the Wuzzuf jobs in Egypt data set:
1.	Read the dataset, convert it to DataFrame and display some from it.
2.	Display structure and summary of the data.
3.	Clean the data (null, duplications)
4.	Count the jobs for each company and display that in order (What are the most demanding companies for jobs?)
5.	Show step 4 in a pie chart.
6.	Find out what are the most popular job titles.
7.	Show step 6 in bar chart.
8.	Find out the most popular areas?
9.	Show step 8 in bar chart.
10.	Print skills one by one and how many each repeated and order the output to find out the most important skills required?
"""
#1
dataset=pd.read_csv('Wuzzuf_Jobs.csv')
dataset

#2
dataset.info()
desc = dataset.describe(include="object")
dataset.nunique()

#3.	Clean the data (null, duplications)

dataset.sort_values("Title",inplace=True)
dataset.drop_duplicates(keep='first',inplace = True)
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,[-1]].values

from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values="",strategy='most_frequent')
x=imputer.fit_transform(x)
dataset.shape

#4.Count the jobs for each company and display that in order (What are the most demanding companies for jobs?)
top_companies=dataset['Company'].value_counts().head()
#5.	Show step 4 in a pie chart.
import matplotlib.pyplot as plt
plt.pie(top_companies,labels=top_companies.keys(),autopct='%1.1f%%')
plt.show()


#6.	Find out what are the most popular job titles.
top_titles = dataset['Title'].value_counts().head()  # Get top titles
#7.	Show step 6 in bar chart.
plt.bar(top_titles.index, top_titles.values)  # Use index for x-axis and values for y-axis
plt.xlabel("Titles")
plt.ylabel("Count")
plt.title("Top 5 Titles Distribution")
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()


#8.	Find out the most popular areas?
top_areas = dataset['Location'].value_counts().head()  # Get top titles
#9.	Show step 8 in bar chart.
plt.bar(top_areas.index, top_areas.values)  # Use index for x-axis and values for y-axis
plt.xlabel("Area")
plt.ylabel("Count")
plt.title("Top 5 Aeas Distribution")
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()


#10.Print skills one by one and how many each repeated and order the output to find out the most important skills required?
skills_list = dataset['Skills'].str.split(',').explode().str.strip().value_counts()
skills_list
































