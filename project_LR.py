import numpy as np
import matplotlib.pyplot as plt
print(np.__version__)

x = [1,2,3,4,5,6]
y = [52,58,66,71,82,94]

hours_x = np.array(x,dtype=float)
scores_y = np.array(y,dtype=float)
'''
print(f'Shape Hours:{hours_x.shape}, Shape Scores:{scores_y.shape}')
plt.scatter(x,y)
plt.title('Scatter of X and Y')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()'''

#split data
train_x = hours_x[:3]
print(train_x)
train_y = scores_y[:3]
print(train_y)
test_x = hours_x[3:]
test_y = scores_y[3:]

#fit line -> calculate b0 and b1
b1 = np.cov(train_x,train_y,bias=True)[0,1]/np.var(train_x) #7
b0 = np.mean(train_y) - b1*np.mean(train_x) #44.67
#y = 7x + 44.67

#evaluation
predictions = 7*test_x +44.67
#print(test_x)
#print(predictions)

MSE = np.mean((test_y - predictions)**2) #20.65
MAE = np.mean(np.abs(test_y - predictions)) #2.66
R2 = 1 - ((np.sum((test_y-predictions)**2))/(np.sum((test_y - np.mean(test_y))**2))) #0.76

x_line = np.linspace(hours_x.min(),hours_x.max(),100)
y_line = b0 +b1*x_line

plt.scatter(x,y)
plt.plot(x_line,y_line,color='purple')
plt.show()












