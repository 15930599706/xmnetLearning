"""
@ coding: UTF-8
@ Model Description：
	这个程序在逻辑回归模型上执行两个不同的逻辑回归实现，一个实现在这个文件中，另一个来自sklearn库。
	目的是：比较两种实现对给定结果的预测效果。最终发现sklearn库中的实现能达到准确率95%以上。
@ Author：DuYanHui
"""

import math
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from numpy import loadtxt, where
from pylab import scatter, show, legend, xlabel, ylabel


def Sigmoid(z):
	G_of_Z = float(1.0 / float((1.0 + math.exp(-1.0*z))))
	return G_of_Z 

def Hypothesis(theta, x):
	z = 0
	for i in range(2):
		z += x[i]*theta[i]
	return Sigmoid(z)

def Cost_Function(X,Y,theta,m):
	sumOfErrors = 0
	for i in range(m):
		xi = X[i]
		hi = Hypothesis(theta,xi)
		if Y[i] == 1:
			error = Y[i] * math.log(hi)
		elif Y[i] == 0:
			error = (1-Y[i]) * math.log(1-hi)
		sumOfErrors += error
	const = -1/m
	J = const * sumOfErrors
	print ('cost is ', J )
	return J

def Cost_Function_Derivative(X,Y,theta,j,m,alpha):
	sumErrors = 0
	for i in range(m):
		xi = X[i]
		xij = xi[j]
		hi = Hypothesis(theta,X[i])
		error = (hi - Y[i])*xij
		sumErrors += error
	m = len(Y)
	constant = float(alpha)/float(m)
	J = constant * sumErrors
	return J

def Gradient_Descent(X,Y,theta,m,alpha):
	new_theta = []
	constant = alpha/m
	for j in range(len(theta)):
		CFDerivative = Cost_Function_Derivative(X,Y,theta,j,m,alpha)
		new_theta_value = theta[j] - CFDerivative
		new_theta.append(new_theta_value)
	return new_theta


def Logistic_Regression(X,Y,alpha,theta,num_iters):
	m = len(Y)
	for x in range(num_iters):
		new_theta = Gradient_Descent(X,Y,theta,m,alpha)
		theta = new_theta
		if x % 100 == 0:
			Cost_Function(X,Y,theta,m)
			print ('theta ', theta)	
			print ('cost is ', Cost_Function(X,Y,theta,m))
	Declare_Winner(theta)

#比较两个模型的准确率
def Declare_Winner(theta):
    score = 0
    # winner = ""

    scikit_score = clf.score(X_test,Y_test)
    length = len(X_test)
    for i in range(length):
        prediction = round(Hypothesis(X_test[i],theta))
        answer = Y_test[i]
        if prediction == answer:
            score += 1
    my_score = float(score) / float(length)
    # if my_score > scikit_score:
    #     print ('自定义模型准确率更高!')
    # elif my_score == scikit_score:
    #     print ('准确率相同!')
    # else:
    #     print( 'Scikit中的模型准确率更高！')
    print ('自定义模型准确率: ', my_score)
    print ('Scikit中的模型准确率: ', scikit_score)



initial_theta = [0,0]
alpha = 0.2
iterations = 15000

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
df = pd.read_csv("score.csv", header=0)

# 数据处理和清洗
cols = [i for i in df.columns if i not in ["label"]]
X = df[cols]
X = np.array(X)
X = min_max_scaler.fit_transform(X)
Y = df["label"]
Y = np.array(Y)

# 数据分割
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33)

# 使用 scikit learn 中的LR模型进行训练
clf = LogisticRegression()
clf.fit(X_train,Y_train)
print ('score Scikit learn: ', clf.score(X_test,Y_test))

Logistic_Regression(X,Y,alpha,initial_theta,iterations)

# 可视化
# pos = where(Y == 1)
# neg = where(Y == 0)
# scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
# scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
# xlabel('Exam 1 score')
# ylabel('Exam 2 score')
# legend(['Not Admitted', 'Admitted'])
# show()

