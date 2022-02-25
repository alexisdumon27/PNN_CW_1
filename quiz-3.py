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

for i in range(50):
    y_labels[i] = 1

for i in range(50, len(y_labels)):
    y_labels[i] = 0

output = LinearRegression().fit(X_data, y_labels).predict(X_data)

print (output)

print (metrics.confusion_matrix(y_labels, output))

