#  𝜃=0.5, 𝑤1=2.5, 𝑤2=−2.5, 𝑤3=−2.5, 𝑤4=2.5

# 𝐻(0)=0.5

import sklearn
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()

X_data = iris.data
y_labels = iris.target

parameter_values = [0.5, 2.5, -2.5, -2.5, 2.5]
learning_rate = 0.01

print (X_data[0])
# 𝐻(0)=0.5 <-- threshold for heavyside

# EPOCH 1 and 2
for i in range(2):
    for i in range(len(X_data)):
        x = X_data[i]
        y = y_labels[i]

        w0 = parameter_values[0]
        w1_x0 = parameter_values[1] * x[0]
        w2_x1 = parameter_values[1] * x[1]
        w3_x2 = parameter_values[1] * x[2]
        w4_x3 = parameter_values[1] * x[3]
        total = w0 + w1_x0 + w2_x1 + w3_x2 + w4_x3

        h = 0
        if total >= 0:
            h = 1
        else:
            h = 0

        for w in range(5):
            if w == 0: 
                parameter_values[w] = parameter_values[w] + learning_rate * (y - h) * 1
            else:
                parameter_values[w] = parameter_values[w] + learning_rate * (y - h) * x[w-1]

print (parameter_values)


# w = w + learning_rate * (target - H(xw)) * x_value_transposed


# EPOCH 2