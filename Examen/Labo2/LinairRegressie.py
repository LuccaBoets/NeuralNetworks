import csv
import math
from statistics import mode

import numpy as np
data = []

csvfile = open('Examen/Labo2/train_fictief.csv', 'r')
csv_reader = csv.reader(csvfile, delimiter=';')
for line in csv_reader:
    data.append([float(line[0]), float(line[1])])

class LinearModel(object):
    def __init__(self, slope, intercept):
        # slope and intercept
        self.slope = slope
        self.intercept = intercept

    def predict(self, x):
        return self.slope * x + self.intercept

    def fit(self, dataset):
        
        sum = 0

        for data in dataset:
            x = data[0]
            y = data[1]

            sum += math.pow(self.predict(x) - y, 2) 

        mean = sum / len(dataset)

        return mean


    def slope_derivative(self, dataset): # 6cost(w,b)/6w
        m = len(dataset)                 # gewoon andere manier om deze te berekenen
        s = 0
        for data in dataset:
            x = data[0]
            y = data[1]

            prediction = self.predict(x)
            s += x * (y - prediction)

        return (-2 / m) * s

    def intercept_derivative(self, dataset): # 6cost(w,b)/6b
        m = len(dataset)                     # gewoon andere manier om deze te berekenen
        s = 0
        for data in dataset:
            prediction = self.predict(data[0])
            s += (data[1] - prediction)

        return (-2 / m) * s

class LinearRegression(object):
    def __init__(self, learningRate, convergenceLimit):
        self.learningRate = learningRate
        self.convergenceLimit = convergenceLimit

    def gradientDescent(self, data):
        
        model = LinearModel(np.random.uniform(), np.random.uniform())
        
        while(True):

            newSlope = model.slope - self.learningRate * model.slope_derivative(data)
            newIntercept = model.intercept - self.learningRate * model.intercept_derivative(data)
            
            if(abs(newSlope - model.slope) < self.convergenceLimit and abs(newIntercept - model.intercept) < self.convergenceLimit):
                model.slope = newSlope
                model.intercept = newIntercept

                break

            else:
                model.slope = newSlope
                model.intercept = newIntercept

        return model
            
lr = LinearRegression(1e-2, 1e-5)

model = lr.gradientDescent(data)

print(model.slope)
print(model.intercept)


# show data as scatter plot
import matplotlib.pyplot as plt
import matplotlib

x = []
y = []

for row in data:
    x.append(float(row[0]))
    y.append(float(row[1]))

plt.scatter(x, y)

# plot model
model_x = np.linspace(0, 10, 100)

model_y = []

for x in model_x:
    model_y.append(model.predict(x))

plt.plot(model_x, model_y, color='g')


plt.show()