import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

hours_x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
score_y = np.array([50, 54, 58, 59, 60, 66,71, 82, 94,94, 96, 97])

label_y = (score_y>=60).astype(int)
#reshape THE X axis, feature should be at least 2D and Y should be exactly 1D
hours_x = hours_x.reshape(-1,1)

#visualization
plt.scatter(hours_x, label_y)
plt.title('X and Y relationship')
plt.xlabel('Hours')
plt.ylabel('Label(pass/fail)')
#plt.show()

#split data
train_x, test_x, train_y, test_y = train_test_split(hours_x, label_y,test_size=0.2,random_state=42)

#train_x, test_x = train_x.reshape(-1,1), test_x.reshape(-1,1)
#print(train_x,'\n')
#print(train_y)

#Logistic regression
log_model = LogisticRegression(solver='lbfgs')
log_model.fit(train_x,train_y)

y_pred = log_model.predict(test_x)
print(classification_report(test_y,y_pred))
#plotting
#plt.plot(hours_x,log_model.predict(hours_x.reshape(-1,1)))
#plt.show()

x_line = np.linspace(hours_x.min(),hours_x.max(),100)
proba = log_model.predict_proba(x_line.reshape(-1,1))[:,1]
plt.plot(x_line,proba)
# The boundary is the value of X where the model switches from FAIL(0) to PASS(1)
x_boundary = -log_model.intercept_[0] / log_model.coef_[0][0]
plt.axvline(x_boundary, color='red')
#plt.axhline(0.5,color='red')
plt.show()

##################
