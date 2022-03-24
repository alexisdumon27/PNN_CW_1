from cProfile import label
from turtle import color
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = [
    [
    [1, 1],
    [1, 2],
    [2, 1],
    [2, 2],
    [3, 2],
    [2, 3],
    [3, 3],
    [4, 3],
    [3, 4],
    [4, 4],],
    [
    [11, 11],
    [11, 12],
    [12, 11],
    [12, 12],
    [13, 12],
    [12, 13],
    [13, 13],
    [14, 13],
    [13, 14],
    [14, 14],],
    [
    [21, 21],
    [21, 22],
    [22, 21],
    [22, 22],
    [23, 22],
    [22, 23],
    [23, 23],
    [24, 23],
    [23, 24],
    [24, 24],]
]

X_0 = [item[0] for item in data[0]]
X_1 = [item[0] for item in data[1]]
X_2 = [item[0] for item in data[2]]
y_0 = [item[1] for item in data[0]]
y_1 = [item[1] for item in data[1]]
y_2 = [item[1] for item in data[2]]

line_0 = []
for i in range(10):
    line_0.append([7, i])

line_1 = []
for i in range(10):
    line_1.append([17, i])

fig = plt.figure()
ax = plt.axes()

plt.scatter(X_0, y_0, color = "green", label = "class 1")
plt.scatter(X_1, y_1, color = "red", label = "class 2")
plt.scatter(X_2, y_2, color = "black", label = "class 3")

ax.set_ylabel('x1')
ax.set_xlabel('x2')
ax.set_title("Data")
plt.legend()
plt.show()
