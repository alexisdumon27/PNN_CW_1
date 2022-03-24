from unittest import result
import sklearn
from sklearn import datasets, metrics
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score

iris = datasets.load_iris()

X_data = iris.data
y_labels = iris.target

# 𝜃=1.4 , 𝑤1=−1.6, 𝑤2=2.7, 𝑤3=−0.5, 𝑤4=1.1,
parameter_values = [-0.4, 2.8, -2.7,-3.2, -1.7]
learning_rate = 0.01

results = []
for i in range(len(X_data)):
    x = X_data[i]
    w1_x0 = parameter_values[1] * x[0]
    w2_x1 = parameter_values[2] * x[1]
    w3_x2 = parameter_values[3] * x[2]
    w4_x3 = parameter_values[4] * x[3]
    total = w1_x0 + w2_x1 + w3_x2 + w4_x3
    
    output = total + 0.4

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

total = 0
for i in range(len(y_labels)):
    if y_labels[i] != results[i]:
        total += 1

error_rate = total / len(y_labels)

print (error_rate)

tn, fp, fn, tp = metrics.confusion_matrix(y_labels, results).ravel()
print (tn, fp, fn, tp)

print (f1_score(y_labels, results))