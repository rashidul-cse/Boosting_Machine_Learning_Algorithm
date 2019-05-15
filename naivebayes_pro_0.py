
import math
import time
import random
import csv


inputfile = 'spambase.csv'

def classify():
    data_set = load_spam_data(inputfile)
    split_ratio = 0.70
    #split the data in the training and test set based on the split ratio
    training_set, test_set = split_data_set(data_set, split_ratio)
    print('Split {0} rows into train={1} and test = {2} rows').format(len(data_set), len(training_set), len(test_set))
    #prepare a summarized model from the trainig set
    summarized_data = summarize_by_class(training_set)
    #Get the classification
    classification = get_classification(summarized_data,test_set)
    accuracy = find_accuracy(test_set,classification)
    print('Accuracy:{0}%').format(accuracy)
    

#Loading spam data from the csv file.
def load_spam_data(filename):
    lines = csv.reader(open(filename, "r"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset
    
#Split data set into training and test set
def split_data_set(dataset, split_ratio):
    training_size = int(len(dataset)* split_ratio)
    training_set = []
    test_set = list(dataset)
    while len(training_set)<training_size:
        index = random.randrange(len(test_set)) #Choosing random rows from the datset and appending them to the training set
        training_set.append(test_set.pop(index))
    return [training_set, test_set]

def segragate_by_class(dataset):
    separated = {}
    for i in range (len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated): #choosing last column as the class value
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated

def mean(numbers):
    return sum(numbers)/float(len(numbers))

#Standard Deviation = Square Root of Variance
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

def summarize(dataset):
    summaries = [(mean(attribute),stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

def summarize_by_class(dataset):
    separated = segragate_by_class(dataset)
    summaries = {}
    for classValue,instances in separated.iteritems():
        summaries[classValue] = summarize(instances)
    return summaries

def calculate_probability(x,mean,stdev): #Gaussian Density Function
    if stdev == 0:
        return 1
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1/(math.sqrt(2*math.pi)*stdev))* exponent

def calculate_class_probability(summaries,inputVector):
    probabilities = {}
    for classValue,classSummaries in summaries.iteritems():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean,stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculate_probability(x,mean,stdev)
        return probabilities

def predict(summaries,inputVector):
    probabilities = calculate_class_probability(summaries,inputVector)
    bestLabel,bestProb = None, -1
    for classValue, probability in probabilities.iteritems():
        if bestLabel is None or probability>bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

def get_classification(summaries,test_set):
    predictions = []
    for i in range(len(test_set)):
        result = predict(summaries,test_set[i])
        predictions.append(result)
    return predictions

def find_accuracy(test_set,predictions):
    correct = 0;
    for x in range(len(test_set)):
        if test_set[x][-1] == predictions[x]:
            correct += 1
        return (correct/float(len(test_set))) *100.0

start_time = time.time()
classify()
print('%s seconds'%(time.time()-start_time))


    
    
    




    

    
