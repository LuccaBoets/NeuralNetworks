import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
import csv

#file = open('C:/Users/Cedric/Documents/GitHub/Python/Neural_Networks/Week1/train_fictief.csv')
data = []

with open("C:/Users/boets/OneDrive/Documents/GitHub/NeuralNetworks/Labo2/train_fictief.csv", 'r') as file:
    csvreader = csv.reader(file, delimiter=';')
    for row in csvreader:
        data.append(row)
data_x = []
data_y = []
data.sort()
for row in data:
    data_x.append(row[0])
    data_y.append(row[1])

#data_x.sort()
#data_y.sort()
print(data)
print("\n\n")
print(data_x)
print(data_y)

#plt.autoscale(False)
plt.scatter(data_x, data_y)
plt.xticks(np.arange(0, 15, step=1),np.arange(0, 15, step=1))

plt.show()