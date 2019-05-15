
import math
import time
import random
import csv

def mean(x):
    return sum(x)/float(len(x))

def stdev(num): 
    return math.sqrt(sum([pow(x-mean(num),2) for x in num])/float(len(num)-1))
   
                   
def nb_summarize(database):
    category = [(mean(tuple), stdev(tuple)) for tuple in zip(*database)]
    del category [-1]
    return category

def calcProbability(x,average,stdev): #Gaussian Density Function
    if stdev == 0:
        return 1
    return (1/(math.sqrt(2*math.pi)*stdev))* (math.exp(-(math.pow(x-average,2)/(2*math.pow(stdev,2)))))

def ham_spam_predict(summaries,inputVector):
    prob= {}
    for cv,cSumm in summaries.iteritems():
        prob[cv] = 1
        for i in range(len(cSumm)):
            mean, stdev = cSumm[i]
            x = inputVector[i]
            prob[cv] *= calcProbability(x, mean, stdev)
    bestLabel, bestProb = None, -1
    for cv, probability in prob.iteritems():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = cv
    return bestLabel

def ham_spam_categorize(nb_summarized_db, test_db):
    ham_spam_res = []
    for i in range(len(test_db)):
        ham_spam_res.append( ham_spam_predict(nb_summarized_db, test_db[i]))
    return ham_spam_res 

        
def findEfficiency(test_db, ham_spam_res ):
    spam_flag= 0;
    for x in range(len(test_db)):
        if test_db[x][-1] == ham_spam_res[x]:
            spam_flag += 1
        return (spam_flag/float(len(test_db)))*100.0
    
start_time = time.time()
file = 'spambase.csv'
tuples= csv.reader(open(file, "rb"))
database=list(tuples)
for i in range(len(database)):
    database[i] = [float(v) for v in database[i]]
train_test_ratio=0.64
train_db=[]
test_db=list(database)
while len(train_db)< int(len(database)*train_test_ratio):
    train_db.append(test_db.pop(random.randrange(len(test_db))))
print('Split {0} rows into train={1} and test = {2} rows').format(len(database), len(train_db), len(test_db))
divided={}
for i in range(len(train_db)):
    tuple = train_db[i]
    if (tuple[-1] not in divided):
        divided[tuple[-1]]=[]
    divided[tuple[-1]].append(tuple)
nb_summarized_db={}

for cv, instance in divided.iteritems():
    nb_summarized_db[cv]=nb_summarize(instance)
ham_spam_res=ham_spam_categorize(nb_summarized_db,test_db)
print('Accuracy:{0}%').format(findEfficiency(test_db, ham_spam_res))
print('%s seconds'%(time.time()-start_time))













    









    
    
    




    

    
