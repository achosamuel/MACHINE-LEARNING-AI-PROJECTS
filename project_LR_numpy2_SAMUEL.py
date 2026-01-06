import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
df = pd.read_csv('BostonHousing.csv')
#show first row
#print(df[:1])
#show the last row
#print(df[-1:])
# show shape
#print(df.shape,'\n')
#general info
#print(df.summarise)

#Data cleaning
#print(df.isnull().sum(),'\n')
#duplicate value
#print(df.duplicated().sum())

#summary statistics
#print(df.summary())

#define the Feature data and the predictor data
Features = df[df.columns[:-1]].values
outcome = df[df.columns[-1]].values

#split data #random_state is to shuffle the data set seed()
train_x,test_x,train_y,test_y = train_test_split(Features,outcome,test_size=0.8,random_state=42)

print(train_x[2])
#visualize the data
## -- Scatter plot
'''
for col in df.columns[:-1]:
    plt.scatter(df[col],df[df.columns[-1]].values)
    plt.title(f'Scatter of {col} and Y')
    plt.xlabel('Predictions')
    plt.xlabel(col)
    #plt.show()'''

## -- Histogram
"""plt.hist(df[df.columns[-1]].values,color='skyblue',density=True)
plt.title('House prices Histogram')
plt.xlabel('Prices')
plt.ylabel('Frequencies')
plt.show()"""

## -- Boxplot
'''plt.boxplot(df[df.columns[-1]].values)
plt.title('House prices Boxplot')
plt.xlabel('Prices')
plt.ylabel('Frequencies')
plt.show()
#print(Features,'\n')
#print(outcome)'''

# Linear regression
lr_model = LinearRegression()
lr_model.fit(train_x,train_y)

y_pred = lr_model.predict(test_x)
print(mean_squared_error(test_y,y_pred)) #25.5640

#plot regression Linear of my model
plt.scatter(test_y,y_pred,color='skyblue')
plt.plot([test_y.min(),test_y.max()],[y_pred.min(),y_pred.max()])
plt.xlabel('Features')
plt.ylabel('Predictions')
plt.title('Linear regression plot')
plt.show()