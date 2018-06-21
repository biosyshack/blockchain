from poloniex import Poloniex
polo = Poloniex()

import numpy as np
import time
import datetime

start = datetime.date(2017,12,1)
stop = datetime.date(2017,12,10)

unixtime_start = time.mktime(start.timetuple())
unixtime_stop = time.mktime(stop.timetuple())

chartdata = polo.returnChartData(currencyPair = 'USDT_BTC', period=300,start = unixtime_start, end=unixtime_stop)

balance = 100

avglist = list()

for i in range(0,len(chartdata)):
	avglist.append(chartdata[i]['weightedAverage'])

npa = np.asarray(avglist, dtype=np.float32)

#np.savetxt('USDT_REP_1712_1709_300.txt',npa)

train_size = int(len(npa) * 0.7)
test_size = len(npa) - train_size
train, test = npa[0:train_size], npa[train_size:len(npa)]

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

print(testX.shape)
