import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score

data = pd.read_csv('magnus_carlsen_games.csv')
#print(data.head)
data['date'] = pd.to_datetime(data['date']) #convert date column to datetime type
data['month_of_playing'] = data['date'].dt.month #extract the month
#print(data['month_of_playing'].head)

data = data.drop(columns=['id','player_name','opponent_name','date','opponent_color']) #game id is not a feature just a label

## encoding of categorical columns
data['format'] = data['format'].map({'Blitz': 0, 'Rapid': 1})
data['result'] = data['result'].map({'Win':1,'Draw':0.5,'Loss':0})
data['player_color'] = data['player_color'].map({'white':0,'black':1})

#re-arange the data columns, put the target at the end
data = data[['player_rating', 'opponent_rating', 'format', 'year',
       'player_color', 'month_of_playing', 'result']]

## creation of our arrays
features = data[data.columns[:-1]].values
target = data[data.columns[-1]].values

x = np.array(features,dtype=float)
y = np.array(target,dtype=float)

## SPLIT DATA TRAIN/TEST
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.2,random_state=42)

#train and test our model
LR_model = LinearRegression()
LR_model.fit(train_x,train_y)
y_predict = LR_model.predict(test_x)

#evaluate our model
MSE = mean_squared_error(test_y,y_predict)
RMSE = root_mean_squared_error(test_y,y_predict)
R_squared = r2_score(test_y,y_predict) #0.049
print(R_squared)
