from loadData import *
from fullClassify import *
# testing()

class LogisticAlgMod(LogisticAlg):

	#Class function that computes the distance comparing a data point to a training point
	#will skip values that are -1
	def computeDistance(self, dataPoint, trainPoint, indicatorVect):
		distance = 0
		# print "indicatorVect was: " + str(indicatorVect)
		indicatorVect = self.modifyIndicatorVect(indicatorVect, dataPoint)
		# print "indicatorVect now is: " + str(indicatorVect)
		for i in range(len(dataPoint)-1):
			if dataPoint[i] == -1:
				continue
			distance += indicatorVect[i] * (dataPoint[i]-trainPoint[i])**2
		distance = abs(distance)**0.5
		return distance

	#Class function that runs the kNearesetNeighbors algorithm on a single datapoint
	def kNearestNeighbors(self, dataPoint, trainData, numNeighbors, indicatorVect):
		distancePoint = collections.namedtuple('distancePoint', 'kvalue cdistance')
		neighborsClassDict = {} #keeps track of the number of classifications of a specific type
		classValuesDict = {}  #Maps distances to grades
		neighborValues = []   #Contains all of the neighbor values
		currentMax = arbitraryMax #sets the ceiling for the distance in the set of data
		i = 0 		#indicates the current number of neighbors
		for k in range(len(trainData)):
			currentNeighbor = trainData[k]
			currentClassify = currentNeighbor[len(currentNeighbor)-1]
			distance = self.computeDistance(dataPoint, currentNeighbor, indicatorVect)
			tempPoint = distancePoint(kvalue = k, cdistance = distance)

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
		return neighborsClassDict, classValuesDict, neighborValues

	# Optimal classification acts as a tie breaker between two grades if they have the same number
	# neighbors
	def optimalClassifictation(self, classValues, neighborValues, grade1, grade2):
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

	# for a singular data point finds the nearest Neighbor, and returns the corresponding grade 
	def findNearest(self, dataPoint, trainData, numNeighbors, indicatorVect):
		classDict = {} #keeps track of the number of classifications of a specific type
		classValues = {}  #Maps datasetNum to grades
		neighborValues = []   #Contains all of the neighbor values

		classDict, classValues, neighborValue = self.kNearestNeighbors(dataPoint, trainData, numNeighbors, indicatorVect)
		maxMatch = 0
		maxGrade = ''
		for j in range(len(classDict.keys())):
			currentGrade = classDict.keys()[j]
			value = classDict[currentGrade]
			if value > maxMatch:
				maxMatch = value
				maxGrade = currentGrade
			if value == maxMatch and maxMatch != 0:
				indicator = self.optimalClassifictation(classValues, neighborValues, currentGrade, maxGrade)
				if indicator == 1:
					maxGrade = currentGrade
			# 	tempDistSum1 = sum(classDict[currentKey])
			# 	tempDistSum2 = sum(classDict[maxGrade])
				# if tempDistSum1 == tempDistSum2:
		# print maxGrade
		return maxGrade

	# runs the classifier through the test data, creating a list 
	def runClassifier(self, trainData, testData, indicatorVect, gradeVector, bound):
		numNeighbors = 5
		classifications = []
		for i in range(len(testData)):
			classifiedGrade = self.findNearest(testData[i], trainData, numNeighbors, indicatorVect)
			classifications.append(classifiedGrade)

		# print classifications
		return classifications

	#based upon the classified grade, returns 1 if it is in a given bound or 0 otherwise
	def checkBounds(self, actualGrade, classifiedGrade,gradeVector, bound):
		actualIndex = 0
		classifiedIndex = 0
		for j in range(len(gradeVector)):
			if gradeVector[j] == classifiedGrade:
				classifiedIndex = j
			if gradeVector[j] == actualGrade:
				actualIndex = j
		return 1 if abs(actualIndex - classifiedIndex) <= bound else 0 

	#checks the classificaiton for each set of data
	def checkClassifications(self, classifications, testData, gradeVector, bound):
		dataLen = len(testData[0])
		numCorrect = 0
		numTotal = 0
		for i in range(len(classifications)):
			classification = classifications[i]
			actual = testData[i][dataLen-1]
			print "classification is: " + str(classification)
			print "actual is: " + str(actual)

			update = checkBounds(actual, classification, gradeVector, bound)
			print "update is: " + str(update)
			numCorrect += update

			numTotal += 1

		probability = float(numCorrect)/numTotal
		print "The overall probability is: " + str(probability)
		return probability

	#modifies the indicator vector based upon how much data is present in each data point
	def modifyIndicatorVect(self, indicatorVect, dataPoint):
		indicatorSum = 0
		for i in range(len(indicatorVect)):
			if dataPoint[i] == -1:
				continue
			indicatorSum += indicatorVect[i]
		newVector = []
		
		for i in range(len(indicatorVect)):
			nIndication = indicatorVect[i]/indicatorSum
			newVector.append(nIndication)
		# print newVector
		return newVector


#Runs a test based upon how much data is present 
#returns the probability o
def partialTest():
	trainData, testData, gradeKeys = reloadData()
	test = LogisticAlgMod(trainData, gradeVector, indicatorVect, 1)
	test.loadData()
	trainData, testData, gradeKeys = reloadData()
	newTest = []
	tempVect = []
	for i in range(len(testData)):
		trial = testData[i][:]
		for j in range(len(tempVect)):
			trial[tempVect[j]] = -1
		newTest.append(trial)
	print newTest[0]
	classifications = test.runClassifier(trainData, newTest, indicatorVect, gradeVector, 1)
	return test.checkClassifications(classifications, testData, gradeVector, 1)

#calls partialTest and runs the number of iterations, and averages the overall probability 
def rerunPartial(numRuns):
	probabilitySum = 0
	for i in range(numRuns):
		probabilitySum += partialTest()
	print probabilitySum
	overall = float(probabilitySum)/numRuns
	print "The average probability is: " + str(overall)


#how much each grade should be weighted based upon Logistic regression 
indicatorVect = [0.07624999981754536, 0.4609230444869674, 0.41210744188031684, 0.2737352723995422, 0.566344024347642, 0.1967150737993122, 0.39516353502779655, 0.5298984496717364, 0.6482926909045612]


rerunPartial(100)