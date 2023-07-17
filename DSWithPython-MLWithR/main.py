#Created by : Amrit Chhetri
# Configure Spark Envronment to run it....

from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
# Load training data
sparkSxn = SparkSession.builder.appName('Linear Regression....').getOrCreate()
print("Session:", sparkSxn)
trainingData = sparkSxn.read.format("libsvm")\
    .load("MSFT")
linearRxn = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
# Training Model
linearRxnModel = linearRxn.fit(trainingData)
# Knowing Intercept
print("Intercept: %s" % str(linearRxnModel.intercept))
# Metrics Summary
trainingSummary = linearRxnModel.summary

print(trainingSummary.totalIterations, trainingSummary.rootMeanSquaredError, trainingSummary.labelCol )
