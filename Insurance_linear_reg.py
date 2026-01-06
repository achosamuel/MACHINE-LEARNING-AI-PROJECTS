import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('insurance.csv')
data['sex'] = (data['sex']=='male').astype(int)
data["smoker"] = (data["smoker"]=='yes').astype(int)
#print(set(data['region'].values)) smoker column is categorical and non binary
#{'southeast', 'northeast', 'southwest', 'northwest'}
#we will use a one hot encode
data['southeast'] = (data['region']=='southeast').astype(int)
data['northeast'] = (data['region']=='northeast').astype(int)
data['southwest'] = (data['region']=='southwest').astype(int)
#we don't create a col for the last category, if all the first third is 0 that means is northwest
#print(data.head) #visualize the data

#we remove col region
#data = data.drop(columns='region')
data = data[['age', 'sex', 'bmi', 'children', 'smoker', 'southeast',
       'northeast', 'southwest', 'charges']]
#i put the target at the end to extract that easily
#print(data.columns)

## Create the feature and target array
features = data[data.columns[:-1]].values
target = data[data.columns[-1]].values
x = np.array(features,dtype=float)
y = np.array(target,dtype=float)
print(x.shape)
#data split
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.2,random_state=42)

#rain and fit our model and test
LR_model = LinearRegression()
LR_model.fit(train_x,train_y)
y_predict_test = LR_model.predict(test_x)

#our model parameters
intercept = LR_model.intercept_ #-12301.896
slopes = LR_model.coef_ #array of the slopes of all features
print(intercept, slopes)
#evaluate our model
MSE = mean_squared_error(test_y,y_predict_test)
RMSE = root_mean_squared_error(test_y,y_predict_test)
R_squared = r2_score(test_y,y_predict_test) #0.783
#print(R_squared)
