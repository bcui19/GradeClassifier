from loadData import *

import collections 
import math

class LogisticAlg(object):

	def __init__(self, testData, grades, indicatorVect, bound):
		self.grades = grades
		self.testData = testData
		# self.indicatorVect = indicatorVect
		self.bound = bound

	def loadData(self):
		trainData, testData, gradeKeys = reloadData()
		self.trainData = trainData
		self.testData = testData
		self.gradeKeys = gradeKeys

	def initializeTotal(self, trainData):
		self.sum = {} # sum of all of the items per a key
		self.total = {} #total number of items per a key
		for i in range(len(self.gradeKeys)):
			self.sum[self.gradeKeys[i]] = [0] * (len(trainData[0])-1)
			self.total[self.gradeKeys[i]] = 0

	def getAverage(self, trainData):
		self.average = {}
		for i in range(len(trainData)):
			currPoint = trainData[i]
			currGrade = currPoint[len(currPoint)-1]
			correspPoint = self.sum[currGrade]
			for j in range(len(currPoint)-1):
				correspPoint[j] += currPoint[j]
			self.sum[currGrade] = correspPoint
			self.total[currGrade] += 1

		for i in range(len(self.sum.keys())):
			currKey = self.sum.keys()[i]
			correspPoint = [float(x)/self.total[currKey] for x in self.sum[currKey]]
			self.average[currKey] = correspPoint
		print self.average

	def getStandardDeviation(self, trainData):
		tempStdev = {}
		for i in range(len(trainData)):
			currPoint = trainData[i]
			currGrade = currPoint[len(currPoint)-1]


	#gets the updatevalue, based upon the difference between the index of the actual grade
	#and the classified grade
	def getUpdateValue(self, bounds, classified, actual):
		actualIndex = 0
		classifiedIndex = 0
		for j in range(len(self.gradeKeys)):
			if actual == self.gradeKeys[j]:
				actualIndex = j
			if classified == self.gradeKeys[j]:
				classifiedIndex = j
		returnValue =  1 - 0.30 * abs(actualIndex - classifiedIndex)
		if returnValue < 0:
			return 0
		else:
			return returnValue

	#Logistic algorithm
	def runLogistic(self, epochMax):
		bounds = 1
		learningRate = 0.00005
		beta = [0] * (len(self.testData[0]))
		gradient = beta[:]
		for epochNum in range(epochMax):
			indicatorVect = beta[1:]
			# indicatorVect = gradient[1:]
			classifiedData = runClassifier(self.testData, self.trainData, indicatorVect, gradeVector, self.bound)

			gradient = [0] * (len(self.testData[0]))
			totalCount = 0
			updateCount = 0
			for index in range(len(self.trainData)):
				# print "dataPoint is: " + str(dataPoint)
				dataPoint = self.trainData[index]
				update = 0
				logValue = self.calcLogValue(dataPoint, beta)
				actualValue = dataPoint[len(self.testData[0])-1] #gets actual classification
				classification = classifiedData[index]
				update = self.getUpdateValue(bounds, classification, actualValue)
				gradient[0] = (update - logValue)
				for j in range(1, len(gradient)):
					currentWeight = float(dataPoint[j-1])/100
					totalCount += 1
					if update != 0:
						updateCount += 1
					gradient[j] += currentWeight * (update - logValue)

			for i in range(len(beta)):
				beta[i] += learningRate * gradient[i]
		print beta
		print "total Count is: " + str(totalCount)
		print "update Count is: " + str(updateCount)
		return beta
	# def returnTrainData(self):
	# 	print trainData

	def calcLogValue(self, dataPoint, beta):
		z = float (beta[0])
		for i in range(1,len(beta)):
			# print "dataPoint is: " + str(dataPoint[i-1])
			# print "beta[i] is: " + str(beta[i])
			z += beta[i] * float(dataPoint[i-1])/100
		# print 1/(1+ math.e**(-z))
		return 1/(1+ math.e**(-z))


def computeDistance(dataPoint, trainPoint, indicatorVect):
	distance = 0
	for i in range(len(dataPoint)-1):
		# print str(trainPoint[i])
		# print str(dataPoint[i])
		# print str(indicatorVect[i])
		distance += indicatorVect[i] * (dataPoint[i]-trainPoint[i])**2
	# print dataPoint[len(dataPoint)-1]
	# print trainPoint[len(trainPoint)-1]
	distance = abs(distance)**0.5
	# print distance
	return distance

def findKNearestNeighbors(dataPoint, trainData, numNeighbors, indicatorVect):
	# print "finding kNearestNeighbors"
	distancePoint = collections.namedtuple('distancePoint', 'kvalue cdistance')

	neighborsClassDict = {} #keeps track of the number of classifications of a specific type
	classValuesDict = {}  #Maps distances to grades
	neighborValues = []   #Contains all of the neighbor values
	currentMax = arbitraryMax #sets the ceiling for the distance in the set of data
	i = 0 		#indicates the current number of neighbord
	for k in range(len(trainData)):
		currentNeighbor = trainData[k]
		currentClassify = currentNeighbor[len(currentNeighbor)-1]
			# print currentClassify
		distance = computeDistance(dataPoint, currentNeighbor, indicatorVect)
		tempPoint = distancePoint(kvalue = k, cdistance = distance)
		# print k
		# print tempPoint

		if i < numNeighbors:
			if len(neighborValues) == 0:
				classValuesDict[k] = currentClassify
				neighborsClassDict[currentClassify] = 1
				neighborValues.append(tempPoint)
				currentMax = distance
			else:
				classifyNum = 0

				if neighborsClassDict.has_key(currentClassify):
					classifyNum = neighborsClassDict[currentClassify]
				classifyNum += 1
				neighborsClassDict[currentClassify] = classifyNum
				classValuesDict[k] = currentClassify
				neighborValues.append(tempPoint)
				neighborValues = sorted(neighborValues, key = lambda distancePoint: distancePoint.cdistance)
				currentMax = neighborValues[len(neighborValues)-1].cdistance

			i += 1

		
		elif distance < currentMax and i >= numNeighbors:
			x = neighborValues.pop().kvalue
			# print x
			y = classValuesDict[x]
			neighborsClassDict[y] -= 1 
			del classValuesDict[x]

			classifyNum = 0
			if not (neighborsClassDict.has_key(currentClassify)):
				neighborsClassDict[currentClassify] = 1
			else:
				neighborsClassDict[currentClassify] += 1


			classifyNum = neighborsClassDict[currentClassify]
			classValuesDict[k] = currentClassify

			neighborValues.append(tempPoint)
			neighborValues = sorted(neighborValues, key = lambda distancePoint: distancePoint.cdistance)
			currentMax = neighborValues[len(neighborValues)-1].cdistance
		# print neighborValues 
		# print ''
		# print ''
	return neighborsClassDict, classValuesDict, neighborValues


def fTupleDistance(neighborValues, key):
	for i in range(len(neighborValues)):
		if neighborValues[i] == key:
			return neighborValues[i].cdistance

#optimal k distance classification based upon the grade
#covers the edge casess of equality 
def optimalKDictClassify(classValues, neighborValues, grade1, grade2):
	grade1Sum = 0
	grade2Sum = 0
	grade1Min = arbitraryMax
	grade2Min = arbitraryMax
	classKeys = classValues.keys()
	for i in range(len(classKeys)):
		currentGrade = classKeys[i]
		if currentGrade == grade1:
			assocDist = fTupleDistance(neighborValues, i)
			grade1Sum += assocDist
			if assocDist < grade1Min:
				grade1Min = assocDist
		elif currentGrade == grade2:
			assocDist = fTupleDistance(neighborValues, i)
			grade2Sum += assocDist
			if assocDist < grade2Min:
				grade2Min = assocDist
	if grade1Sum > grade2Sum:
		return 2
	elif grade2Sum > grade1Sum:
		return 1
	else:
		if grade1Min < grade2Min:
			return 2
		else:
			return 1


#finds the k nearest neighbors for a data point
def kNearestNeighbors(dataPoint, trainData, numNeighbors, indicatorVect):
	# print dataPoint

	classDict = {} #keeps track of the number of classifications of a specific type
	classValues = {}  #Maps datasetNum to grades
	neighborValues = []   #Contains all of the neighbor values
	#return neighborsClassDict, classValuesDict, neighborValues

	classDict, classValues, neighborValue = findKNearestNeighbors(dataPoint, trainData, numNeighbors, indicatorVect)
	# print ''
	# print "classValues" + str(classValues)
	# print "neighborValue" + str(neighborValue)
	# print "classDict" + str(classDict)
	maxMatch = 0
	maxGrade = ''
	for j in range(len(classDict.keys())):
		currentGrade = classDict.keys()[j]
		value = classDict[currentGrade]
		# print "the key: " + str(currentKey) + " has " + str(classDict[currentKey]) + " matches"
		if value > maxMatch:
			maxMatch = value
			maxGrade = currentGrade
		if value == maxMatch and maxMatch != 0:
			indicator = optimalKDictClassify(classValues, neighborValues, currentGrade, maxGrade)
			if indicator == 1:
				maxGrade = currentGrade
		# 	tempDistSum1 = sum(classDict[currentKey])
		# 	tempDistSum2 = sum(classDict[maxGrade])
			# if tempDistSum1 == tempDistSum2:
	# print maxGrade
	return maxGrade


#runs the classifier once
def runClassifier(trainData, testData, indicatorVect, gradeVector, bound):
	# print "len of trainData is: " + str(len(trainData))
	# print "len of testData is: " + str(len(testData))
	numNeighbors = 5
	classifications = []
	for i in range(len(testData)):
		classifiedGrade = kNearestNeighbors(testData[i], trainData, numNeighbors, indicatorVect)
		classifications.append(classifiedGrade)

	return classifications

#checks the classification by consistently reloading the dataset 
def checkClassification(numIterations, indicatorVect, gradeVector, upperBound):
	probSum = 0
	for i in range(numIterations):
		trainData, testData, gradeKeys = reloadData()
		probSum += runClassifier(trainData, testData, indicatorVect, gradeVector, upperBound)

	probability = float(probSum)/numIterations
	# print "The overall probability is: " + str(probability)
	return probability




def updateGoalDict(optimalDict):
	goalKeys = optimalDict.keys()
	minimum = arbitraryMax
	for i in range(len(goalKeys)):
		currentProb = goalKeys[i]
		if currentProb < minimum:
			minimum = currentProb
			
	# del optimalDict[minimum]
	return minimum

# Bash method to verify the bounds of the grades
# def optimizeGradeBounds(indicatorVect):
# 	# gradeKeys = ['A', 'C+', 'C', 'B', 'B-', 'A-', 'B+']
# 	gradeVector = ['A', 'A-', 'B+', 'B', 'B-', 'C+', 'C']
# 	for i in range(4):
# 		tprobability = checkClassification(1000, indicatorVect, gradeVector, i)
# 		print "For the upper bound of:" , i,  "the probability is: ", tprobability

#Runs runs the logistic regression the trialNum number of times 
#and with a given input epoch 
def findBetaValue(epochNum, trialNum):
	trainData, testData, gradeKeys = reloadData()
	y = [0, 0, 1, 1, 1, 0, 1, 1]

	test = LogisticAlg(testData, gradeVector, y, 1)
	betaTotal = [0] * (len(testData[0]) + 1)
	for i in range(trialNum):
		print "The value of i is: " + str(i)
		test.loadData()
		tempBeta = test.runLogistic(epochNum)
		betaTotal = [x + y for x, y in zip(tempBeta, betaTotal)]

	print "betaTotal is: " + str(betaTotal)
	betaAverage = [x/trialNum for x in betaTotal]
	return betaAverage 

def checkBounds(classifiedGrade, actualGrade, gradeVector, bound):
	actualIndex = 0
	classifiedIndex = 0
	for j in range(len(gradeVector)):
		if gradeVector[j] == classifiedGrade:
			classifiedIndex = j
		if gradeVector[j] == actualGrade:
			actualIndex = j
	# return abs(actualIndex - classifiedIndex) <= bound ? 1: 0
	# print "actualGrade is: " + actualGrade
	# print "classifiedGrade is " + classifiedGrade
	x = 1 if abs(actualIndex - classifiedIndex) <= bound else 0
	# print x
	return x

#evaluates the probability that the classifier had a good probability based upon the 
#input dataset and the bound
def checkClassifier(trainData, testData, indicatorVect, gradeVector, bound):
	# print "running classifier"
	numNeighbors = 5
	numTotal = 0
	numCorrect = 0
	vectorLen = len(testData[0])
	#runs through every data point of the testingData
	print "The indicatorVect is: " + str(indicatorVect)
	for i in range(len(testData)):
		numTotal += 1
		dataSet = testData[i]
		classifiedGrade = kNearestNeighbors(dataSet, trainData, numNeighbors, indicatorVect)
		# print "Classified grade is: " + str(classifiedGrade)
		# print "Actual grade is: " + str(testData[i][vectorLen - 1])
		
		actualGrade = dataSet[len(dataSet)-1]
		update = checkBounds(classifiedGrade, actualGrade, gradeVector, bound)

		numCorrect += update


	probability = float(numCorrect)/numTotal
	print "The probability that there was a correct classification is: " + str(probability)
	return probability


# def runKNearestNeighbors(epochNum, trialNum):
# 	# indicatorVect = findBetaValue(epochNum, trialNum)
# 	indicatorVect = [0, 0, 1, 1, 1, 1, 1, 1]


# 	trainData, testData, gradeKeys = reloadData()
# 	checkClassifier(trainData, testData, indicatorVect, gradeVector, 1)

#testing the class workings 
def testing():
	trainData, testData, gradeKeys = reloadData()
	y = [0, 0, 1, 1, 1, 0, 1, 1]

	test = LogisticAlg(trainData, gradeVector, y, 1)
	test.loadData()
	test.initializeTotal(trainData)
	test.getAverage(trainData)
	indicatorVect = findBetaValue(5000, 1)
	trainData, testData, gradeKeys = reloadData()
	checkClassifier(trainData, testData, indicatorVect, gradeVector, 1)




arbitraryMax = 100000
gradeVector = ['A', 'A-', 'B+', 'B', 'B-', 'C+', 'C']
# findBetaValue(1000, 10)
# testing()








