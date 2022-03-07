import sklearn
from sklearn import datasets
import numpy as np

# g(x) > 0 for class 0 and g(x) <= 0 for all other classes

iris = datasets.load_iris()

X_data = iris.data
y_labels = iris.target

# Apply 2 epochs of the Sequential Widrow-Hoff Learning Algorithm
#   𝐚𝑇=(𝑤0,𝐰𝑇)=(0.5,−0.5,−1.5,2.5,−1.5)

# initial values
a = [0.5, -0.5, -1.5, 2.5, -1.5]

# margin vector
b = 1

# learning rate
n = 0.01

# margin vector 𝐛 in which all values are equal to 1, and use a learning rate of 0.01.

# for each sample in the dataset
# update the weight based on the equation
total = 0
for i in range(len(y_labels)):
    x = X_data[i]

    discriminant_output = a[0] * 1 # a^t * 
    discriminant_output += a[1] * x[0]
    discriminant_output += a[2] * x[1]
    discriminant_output += a[3] * x[2]
    discriminant_output += a[4] * x[3]

    if y_labels[i] == 0 and discriminant_output > 0:
        total += 1
    elif discriminant_output <= 0 and y_labels[i] == 1:
        total += 1
    elif discriminant_output <= 0 and y_labels[i] == 2:
        total += 1

print (total / len(y_labels))

print (X_data[0])
for w in range(2):
    for i in range(len(X_data)):
        x = X_data[i]
        y = y_labels[i]
        discriminant_output = a[0] * 1 # a^t * 
        discriminant_output += a[1] * x[0]
        discriminant_output += a[2] * x[1]
        discriminant_output += a[3] * x[2]
        discriminant_output += a[4] * x[3]

        # print (discriminant_output)
        a[0] += n*(b-discriminant_output)*1
        a[1] += n*(b-discriminant_output)*x[0]
        a[2] += n*(b-discriminant_output)*x[1]
        a[3] += n*(b-discriminant_output)*x[2]
        a[4] += n*(b-discriminant_output)*x[3]
        
        # print (a)
        # raise ValueError("stop")

print (a)

total = 0
for i in range(len(y_labels)):
    x = X_data[i]

    discriminant_output = a[0] * 1 # a^t * 
    discriminant_output += a[1] * x[0]
    discriminant_output += a[2] * x[1]
    discriminant_output += a[3] * x[2]
    discriminant_output += a[4] * x[3]

    if y_labels[i] == 0 and discriminant_output > 0:
        total += 1
    elif discriminant_output <= 0 and y_labels[i] == 1:
        total += 1
    elif discriminant_output <= 0 and y_labels[i] == 2:
        total += 1

print (total / len(y_labels))