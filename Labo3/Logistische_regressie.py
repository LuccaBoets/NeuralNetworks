from random import uniform
import matplotlib.pyplot as plt
import csv
import numpy as np
import math as math
from os import environ

def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"


class Model:
    def __init__(self, slope, intercept):
        self.slope = slope
        self.intercept = intercept

    def forward(self, input):
        s = 0
        for i in range(len(self.slope)):
            s += self.slope[i] * input[i]
        return s + self.intercept

    def sigmoid(self, input):
        return 1/(1+math.exp(-model.forward(input)))

class TrainingSample:
    def __init__(self, features, label):
        self.features = features
        self.label = label


class TrainingSet:
    def __init__(self):
       self.training_samples = []  

    def append(self, training_sample):
      self.training_samples.append(training_sample)




trainingSet = TrainingSet()

with open("C:/Users/boets/OneDrive/Documents/GitHub/NeuralNetworks/Labo3/train.csv", 'r') as file:
    csvreader = csv.reader(file, delimiter=';')
    for row in csvreader:
        trainingSet.training_samples.append(TrainingSample([float(row[1]),float(row[2]),float(row[3]),float(row[4]),float(row[5]),float(row[6])],float(row[0]))) 


# print(trainingSet.training_samples[0].features)

minFiness = float("inf")
minModel = None


for i in range(100):

    slope = []
    for j in range(6):
        slope.append(uniform(0,10))

    model = Model(slope, uniform(0,10))

    currentFiness = 0.0

    for training_sample in trainingSet.training_samples:

        

        currentFiness += - (training_sample.label*math.log(model.sigmoid(training_sample.features))) - ((1-training_sample.label) * math.log(1.0001-model.sigmoid(training_sample.features)))

        # if training_sample.label == 1:
        #     currentFiness += - math.log(model.sigmoid(training_sample.features))
        # else:
        #     # print(model.sigmoid(training_sample.features))
        #     currentFiness += - math.log(1.0001-model.sigmoid(training_sample.features))

    currentFiness /= len(trainingSet.training_samples)


    if currentFiness <= minFiness:
        minFiness = currentFiness
        minModel = model

for x in trainingSet.training_samples:
    print(minModel.sigmoid(x.features))



suppress_qt_warnings()

data_x = []
data_y = []

for training_sample in trainingSet.training_samples:
    data_x.append(minModel.sigmoid(training_sample.features))
    data_y.append(training_sample.label)

plt.scatter(data_x, data_y)
# plt.xticks(np.arange(0, 15, step=1),np.arange(0, 15, step=1))

# plt.plot(data_x, data_y, '-r')

plt.show()