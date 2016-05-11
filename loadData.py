'''
loadData.py
------------
From a .csv file loads and parses the data
from a CSV file into python to be made useful for training and testing 
'''


import os
import random
import copy
import math

def loadData(dataFile, spacing, tags):
	rawData = open(dataFile, 'r', 0)
	dataSet = []
	counter = 0

	for line in rawData:
		counter += 1
		if counter < spacing:
			continue

		currentLine = line.split(',')
		processedData = extractData(currentLine, tags)
		dataSet.append(processedData[:])
	return dataSet

def extractData(dataLine, tags):
	sData = []
	for i in range(len(tags)):
		dataPoint = dataLine[tags[i]]
		if i != len(tags)-1:
			if dataPoint == '':
				dataPoint = 0
			else:
				dataPoint = float(dataPoint)

		sData.append(dataPoint)
	return sData

def getInfo(dataFile, spacing, keywords):
	rawData = open(dataFile, 'r', 0)
	info = []
	infoIndex = []
	counter = 0
	for line in rawData:
		counter += 1
		if spacing == counter:
			info = line.split(',')
			break
	
	for i in range(len(info)):
		if info[i] in keywords:
			infoIndex.append(i)

	return infoIndex

def createInitialGradeMap(inputGrades):
	grades = {}
	for i in range(len(inputGrades)):
		grades[inputGrades[i]] = 0
	return grades

def createGradeMap(dataSet, inputGrades):
	grades = createInitialGradeMap(inputGrades)
	dataSetSize = len(dataSet[0])
	for i in range(len(dataSet)):
		currentGrade = dataSet[i][dataSetSize-1]
		if currentGrade in grades.keys():
			grades[currentGrade] += 1
	# for j in range(len(grades.keys())):
		# print "The grade distribution is for " + grades.keys()[j] + " is: " + str(grades[grades.keys()[j]])
	# print grades
	return grades

'''
Removes data that has too few data points to train/test on
Removing extremes from the dataset
'''

def simplifyData(dataSet, grades, threshold, extra):
	remove = extra
	condensedData = []
	for i in range(len(grades.keys())):
		key = grades.keys()[i]
		if (grades[key] < threshold):
			remove.append(key)

	# print remove

	for i in range(len(dataSet)):
		if dataSet[i][dataSetLength-1] in remove:
			continue
		condensedData.append(dataSet[i])
	# print len(condensedData)
	# print len(dataSet)
	return condensedData
'''
Chooses what data to test/train on
done randomly with a given probability 
'''
def chooseData(dataSet, grades, distribution):
	# print random.randint(0, 2)
	train = []
	test = []
	for i in range(len(dataSet)):
		actualP = random.randint(0, 100)
		if float(actualP)/100 < distribution:
			train.append(dataSet[i])
			continue
		test.append(dataSet[i])

	# print "Test Length is: " + str(len(test))
	# print "Train length is: " + str(len(train))
	# print "The ratio is: " + str(float(len(test))/(len(train)+len(test)))
	return train, test


def listToMap(dataSet):
	gradeMap = {}
	for i in range(len(dataSet)):
		# print dataSet[i][len(dataSet[0])-1]
		currentKey = dataSet[i][len(dataSet[0])-1]
		if gradeMap.has_key(currentKey):
			currentList = gradeMap[currentKey]
			# print currentList
			currentList.append(dataSet[i])
			gradeMap[currentKey] = currentList
		else:
			currentList = []
			currentList.append(dataSet[i])
			gradeMap[currentKey] = currentList
	# print gradeMap
	return gradeMap.keys()




def reloadData():
	global dataSetLength, globalBound, lGrades
	directory = os.path.dirname(__file__)

	filename = os.path.join(directory, 'CS-110-Win-1516-Grades-Modified.csv')
	keywords = ['Adjusted', 'Curved', 'Grade']
	# keywords = ['Adjusted', 'Grade']
	inputGrades = ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'D-']
	extra = ['NC', 'I', '', ' ']

	dataSetLength = 0
	arbitraryLength = 50000

	tags = getInfo(filename, 3, keywords) #from the .csv file allows us to get the relevant data
	# print tags 
	dataSet = loadData(filename, 5, tags)
	dataSetLength = len(dataSet[0])
	grades = createGradeMap(dataSet, inputGrades)
	lGrades = grades.keys()

	condensedData = simplifyData(dataSet, grades, 6, extra)
	trainData, testData = chooseData(condensedData, grades, 0.7)


	# print trainData
	# print testData
	gradeKeys = listToMap(condensedData)
	# print gradeKeys

	return trainData, testData, gradeKeys



# reloadData()



