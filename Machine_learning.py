import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import tensorflow as tf

col_names = ['fLength','fWidth','fSize','fConc','fConc1','fAsym','fM3Long','fM3Trans',
             'fAlpha','fDist','class']

df = pd.read_csv('magic04.data',names=col_names)
#print(df['class'].unique())
#print(df.shape)
#print((df['class']!='g').sum())
#print(df.head())

df['class'] = (df['class']=='g').astype(int)
#print(df.head())
#print(df.shape)
#print((df['class']=='g').sum())
"""
for label in col_names[:-1]:
    plt.hist(df[df['class']==1][label],color='blue',alpha=0.7,density=True,label='gamma')
    plt.hist(df[df['class']==0][label],color='red',alpha=0.7,density=True,label='hadron')
    plt.title(f'Histogram of {label}')
    plt.ylabel(label)
    plt.xlabel('Frequency')
    plt.legend()
    plt.show()"""
#print(df[df.columns[:-1]])
#train, valid, test
train, valid, test = np.split(df.sample(frac=1),[int(0.6*len(df)),int(0.8*len(df))])

def scale_dataset(dataframe, oversample=False):
    x = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    if oversample:
        ros = RandomOverSampler()
        x,y = ros.fit_resample(x,y)

    data = np.hstack((x,np.reshape(y,(-1,1))))

    return data, x, y

train, x_train, y_train = scale_dataset(train,True)
valid, x_valid, y_valid = scale_dataset(valid,False)
test, x_test, y_test = scale_dataset(test,False)

#print(len(train[train['class']==1])) #GAMMA :7304
#print(len(train[train['class']==0])) #HANDRA : 4108
#gamma data is much greater than handrana data
# so we gonna import imblearn.over_sampling -> RandomOverSampler

knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(x_train,y_train)

y_pred = knn_model.predict(x_test)
#print(y_pred)
print(classification_report(y_test,y_pred),'\n')

######################3 naives bayes
nb_model = GaussianNB()
nb_model.fit(x_train,y_train)
y_pred1 = nb_model.predict(x_test)

print(classification_report(y_test,y_pred1),'\n')

############Logistic regression
log_model = LogisticRegression()
log_model.fit(x_train,y_train)
y_pred2 = log_model.predict(x_test)

print(classification_report(y_test,y_pred2),'\n')

#### Support vector machine (SVM)
svm_model = SVC()
svm_model.fit(x_train,y_train)

y_pred3 = svm_model.predict(x_test)
print(classification_report(y_test,y_pred3),'\n')

##### Neural Networks












