import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error



def main():
    dataset = pandas.read_csv('C:\\Users\\boets\\OneDrive\\Documents\\\GitHub\\NeuralNetworks\\Labo10/airline-passengers.csv', usecols=[1], engine='python')
    plt.plot(dataset)

    numpy.random.seed(7)
    dataframe = pandas.read_csv('C:\\Users\\boets\\OneDrive\\Documents\\GitHub\\NeuralNetworks\\Labo10/airline-passengers.csv', usecols=[1], engine='python')
    dataset = dataframe.values
    dataset = dataset.astype('float32')

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    train_size = int(len(dataset) * 0.7)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

    print(len(train), len(test))

    trainMatrixX, trainMatrixY = create_matrix(train)
    testMatrixX, testMatrixY = create_matrix(test)
    trainMatrixX = numpy.reshape(trainMatrixX, (trainMatrixX.shape[0], 1, trainMatrixX.shape[1]))
    testMatrixX = numpy.reshape(testMatrixX, (testMatrixX.shape[0], 1, testMatrixX.shape[1]))

    network = Sequential()
    network.add(LSTM(8, input_shape=(1, 1)))
    network.add(Dense(2))
    network.add(Dense(1))
    network.compile(loss='mean_squared_error')

    network.fit(trainMatrixX, trainMatrixY, epochs=10, batch_size=1, verbose=2)

    trainPredict = network.predict(trainMatrixX)
    testPredict = network.predict(testMatrixX)

    trainPredict = scaler.inverse_transform(trainPredict)
    trainMatrixY = scaler.inverse_transform([trainMatrixY])

    testPredict = scaler.inverse_transform(testPredict)
    testMatrixY = scaler.inverse_transform([testMatrixY])

    trainScore = math.sqrt(mean_squared_error(trainMatrixY[0], trainPredict[:,0]))
    print('Train Score: '+str(trainScore)+' RMSE')

    testScore = math.sqrt(mean_squared_error(testMatrixY[0], testPredict[:,0]))
    print('Test Score: '+str(testScore)+' RMSE')

    plotData1 = numpy.empty_like(dataset)
    plotData1[:, :] = numpy.nan
    plotData1[1:len(trainPredict)+1, :] = trainPredict

    plotData2 = numpy.empty_like(dataset)
    plotData2[:, :] = numpy.nan
    plotData2[len(trainPredict)+(1*2)+1:len(dataset)-1, :] = testPredict

    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(plotData1)
    plt.plot(plotData2)

    plt.show()

def create_matrix(dataset, look_back=1):
    x = []
    y = []

    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        x.append(a)
        y.append(dataset[i + look_back, 0])
    return numpy.array(x), numpy.array(y)

if __name__ == "__main__":
    main()
