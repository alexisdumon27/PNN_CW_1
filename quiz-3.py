from unittest import result
import sklearn
from sklearn import datasets, metrics
import numpy as np
from sklearn.linear_model import LinearRegression


iris = datasets.load_iris()

X_data = iris.data
y_labels = iris.target

# 𝜃=1.4 , 𝑤1=−1.6, 𝑤2=2.7, 𝑤3=−0.5, 𝑤4=1.1,
parameter_values = [1.4, -1.6, 2.7, -0.5, 1.1]
learning_rate = 0.01

results = []
for i in range(len(X_data)):
    x = X_data[i]
    w1_x0 = parameter_values[1] * x[0]
    w2_x1 = parameter_values[2] * x[1]
    w3_x2 = parameter_values[3] * x[2]
    w4_x3 = parameter_values[4] * x[3]
    total = w1_x0 + w2_x1 + w3_x2 + w4_x3
    
    output = total - 1.4
    if output > 0:
        results.append(1)
    else:
        results.append(0)

for i in range(50):
    y_labels[i] = 1

for i in range(50, len(y_labels)):
    y_labels[i] = 0

# output = LinearRegression().fit(X_data, y_labels).predict(X_data)

# print (output)

print (metrics.confusion_matrix(y_labels, results))

