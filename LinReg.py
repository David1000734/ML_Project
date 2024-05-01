import pandas as pd
import numpy
from math import sqrt


def main():

    # load in data and start pre-procesing it
    BaseData = pd.read_csv('diabetes_binary_5050split_health_indicators_.csv', header=None)    # import data into dataframe
    BaseData = BaseData.iloc[1:]  # strips headers
    BaseData = BaseData.astype(float)                       # convert data to floats for future operations
    BaseData = BaseData.sample(frac=1)                      # shuffle data into random order
    BaseData = BaseData.reset_index(drop=True)              # reset indices
    BaseData = BaseData.drop(BaseData.index[:92]).reset_index(drop=True)  # drop first sample and reset indices

    # print(len(BaseData))
    # print(BaseData)
    print('\t\t\tRMSE')


    # CV - manual implementation
    numFolds = 100
    fSize = len(BaseData) // numFolds       # determine fold size
    for i in range(0, numFolds):
        test = BaseData.iloc[i*fSize:(i+1)*fSize].copy().reset_index(drop=True)
        train = BaseData
        train = train.drop(train.index[i*fSize:(i+1)*fSize]).reset_index(drop=True)

        #Z-Score Normalization
        #for each training column except first convert values to z-score
        #then use training std and mean to convert test data to z-scores
        for j in range(1, 22):
            xmean = train[j].mean()
            xstd = train[j].std()

            train[j] = (train[j] - xmean) / xstd
            test[j] = (test[j] - xmean) / xstd


        # Regression Calculations
        VectorX = train.drop(columns=0) #drop diabetes column from X
        VectorX = pd.DataFrame(numpy.c_[numpy.ones(len(VectorX)), VectorX])     # add ones column to X
        VectorY = train.drop(columns=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])   #drop everything but diab from Y

        #Perform the same data processing on test data sets
        testX = test.drop(columns=0)
        testX = pd.DataFrame(numpy.c_[numpy.ones(len(testX)), testX])
        testY = test.drop(columns=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
        # print(VectorY)
        # print(VectorX)

        #calculate b vector with the formatted training data
        b = numpy.dot(VectorX.transpose(), VectorX)
        b = numpy.linalg.inv(b)
        b = numpy.dot(b, VectorX.transpose())
        b = numpy.dot(b, VectorY)
        #print(b)

        #Calculate Y vector using test data and coefficient s
        CalcY = numpy.dot(testX, b)
        PredictY = pd.DataFrame(CalcY)
        #print(PredictY)
        #print(VectorY)


        #Calculate RMSE
        difsq = (PredictY - testY)**2
        sums = difsq.sum()
        RMSE = sqrt(sums[0]/fSize)

        # print(PredictY)
        # print(PredictY[0][0])
        # print(PredictY[0][1])

        # convert regression values into actual values
        for num in range(0, len(PredictY)):
            if PredictY[0][num] > 0.5:
                PredictY.loc[num, 0] = 1
            else:
                PredictY.loc[num, 0] = 0

        #print(PredictY)

        tSum = 0
        for num in range(0, len(PredictY)):
            if PredictY[0][num] == testY[0][num]:
                tSum += 1

        # print(tSum)
        acc = "Accuracy is: " + str(tSum / len(PredictY)) + '\n'
        #print(acc)


        #print table results for current fold
        if (i+1) % 10 == 0:
            if i != 99:
                output = "Fold " + str(i + 1) + '\t\t'
            else:
                output = "Fold " + str(i + 1) + '\t'
            output += "{:.4f}".format(RMSE)
            print(output)
            print(acc)

#run main
main()
