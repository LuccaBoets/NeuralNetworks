from random import uniform
import matplotlib.pyplot as plt
import csv
import numpy as np
import math as math

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


for i in range(1):

    slope = []
    for j in range(6):
        slope.append(uniform(0,10))

    model = Model(slope, uniform(0,10))

    currentFiness = 0.0

    for training_sample in trainingSet.training_samples:
        print(training_sample.label)

        currentFiness += (training_sample.label*math.log(model.sigmoid(training_sample.features))) + ((1-training_sample.label) * math.log(1-model.sigmoid(training_sample.features)))

        # if training_sample.label == 1:
        #     currentFiness += - math.log(model.sigmoid(training_sample.features))
        # else:
        #     print(model.sigmoid(training_sample.features))
        #     currentFiness += - math.log(1-model.sigmoid(training_sample.features))


    print(currentFiness/ len(trainingSet.training_samples))
