from random import uniform
import matplotlib.pyplot as plt
import csv
import numpy as np

class Model:
    def __init__(self, slope, intercept):
        self.slope = slope
        self.intercept = intercept
    
            

data = []

with open("C:/Users/boets/OneDrive/Documents/GitHub/NeuralNetworks/Labo2/train_fictief.csv", 'r') as file:
    csvreader = csv.reader(file, delimiter=';')
    for row in csvreader:
        data.append([float(row[0]), float(row[1])])

minFiness = float("inf")
minModel = None

learningRate = 1e-2
convergenceLimit = 1e-5

model = Model(uniform(0,10), uniform(0,10))

for x in range(100):

# while True:

    # for row in data:
    #     currentFiness += (row[1] - (model.slope*row[0]+model.intercept))

    # # (-2 / len(training_set.training_samples)) * s

    # newIntersept = model.intercept - learningRate * 
    # newSlope = 0

    currentFiness = 0.0

    for row in data:
        currentFiness += (row[1] - (model.slope*row[0]+model.intercept))**2

    currentFiness /= len(data)

    print(currentFiness)

    if currentFiness < minFiness:
        minFiness = currentFiness
        minModel = model


data_x = []
data_y = []

for row in data:
    data_x.append(row[0])
    data_y.append(row[1])

print("minFiness")
print(minFiness)
print(minModel.slope)
print(minModel.intercept)


plt.scatter(data_x, data_y)
plt.xticks(np.arange(0, 15, step=1),np.arange(0, 15, step=1))

x = np.linspace(-1,10,100)
y = minModel.slope*x+minModel.intercept
plt.plot(x, y, '-r')


# plt.plot(yPoint, linestyle = 'dotted')

plt.show()