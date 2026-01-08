import numpy as np
import matplotlib.pyplot as plt

hours = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
score =  [50, 54, 58, 59, 60, 66,71, 82, 94,94, 96, 97]

x = np.array(hours,dtype=int)
score_array = np.array(score,dtype=int)
y = (score_array>=60).astype(int)

#quick visualization
plt.figure()
plt.scatter(x,y)
plt.xlabel('Hours studied')
plt.ylabel('1 : Pass / 0 :Fail')
#plt.show()

#scale our features
#Scaling before splitting is wrong because the mean and std are learned parameters,
# and learning anything from the test set violates the idea of “unseen data”
#x = (x-x.mean())/x.std(ddof=0)
#print(x)

#Split our data TRAIN/TEST
rng = np.random.default_rng(42)
index_ = np.arange(len(x))
rng.shuffle(index_)

pos = int(len(x)*0.8)
train_id,test_id = index_[:pos], index_[pos:]

train_x,train_y = x[train_id], y[train_id]
test_x,test_y = x[test_id], y[test_id]
## Features scaling
mean_ = train_x.mean()
std_ = train_x.std(ddof=0)
train_x = (train_x-mean_)/std_
test_x = (test_x-mean_)/std_
print(train_x,test_x)
## Our model fitting, z = w*x +b, p=sigmoid(z) = 1/(1+exp(-z))
def sigmoid(z):
    return 1/(1+np.exp(-z))

w,b = 0,0
alpha_ = 0.2 #learning rate
m = len(train_x)
iterations = 3000 #nb of repetitions
cost_list = []

for _ in range(iterations):
    z = w*train_x + b
    p = sigmoid(z)
    ##
    epsilon = 1e-15
    np.clip(p,epsilon, 1-epsilon)

    #cost (loss) functions
    cost = -(1/m)*np.sum(train_y*np.log(p)+(1-train_y)*np.log(1-p))
    cost_list.append(cost)
    #gradien descent
    dw = (1/m)*np.sum((p-train_y)*train_x)
    db = (1/m)* np.sum(p-train_y)

    w -= alpha_*dw
    b -= alpha_*db
print(f'w:{w},b:{b}')
# our model : p = sigmoid(wx+b)
#w:9.607 ; b:4.992
# Model testing
p_predict = sigmoid(w*test_x+b) #[0.9999999  0.00178719 0.99998534]
#print(p_predict)
#class (threshold = 0.5)
p_predict = (p_predict>=0.5).astype(int) #[1 0 1]
#print(p_predict)

#accuracy metrics
acc = (p_predict==test_y).mean() #100%
#print(acc)

## Plotting
plot_x = np.linspace(x.min()-0.5,x.max()+0.5,100)
plot_y = sigmoid(w*plot_x+b)
#plot_y = (plot_y>=0.5).astype(int)

x_star = None
if w > 1e-12:
    x_star = -b/w

plt.figure()
plt.scatter(train_x,train_y,color='brown')
plt.xlabel('Deviation form the mean')
plt.ylabel('Fail:0, Pass:1')
plt.plot(plot_x,plot_y,color='red')
if x_star is not None and x.min()<= x_star <= x.max():
    plt.axvline(x_star,color='darkred')
#plt.show()

