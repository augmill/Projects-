import random
import numpy as np #np or torch?
import pandas as pd
from tqdm.autonotebook import tqdm

"""function goes through all the data and processes it by finding where in the setence the keyword is and 
replacing it with the target symbol * and adds a tuple of the processed sentence and the keyword. a set 
is used to prevent duplicate datum. takes the data"""
def processing(inData, corpus):
    # the data to be returned 
    outData = []
    # the set to prevent duplicates
    dataSet = set()
    for sentence, info, keyword in inData:
        # finds the words POS
        # pos = coca.search(info[1], info[2], info[3], fileName=info[0]).split(' ')[-1]
        if 'v' not in corpus.search(info[1], info[2], info[3], fileName=info[0]).split(' ')[-1]:
            continue
        #sentence split up
        splitSent = sentence[0].split(' ')
        # [the index is the keyword's index which is given as part of the kwic search]
        splitSent[info[3]] = '*'
        fullSent = ' '.join(splitSent)
        if (fullSent, keyword) not in dataSet:
            dataSet.add((fullSent, keyword))
            # sentence, sentence info, and sentence keyword
            outData.append((fullSent, info, keyword)) # , pos
    return outData

# function divides up data into training, dev, and test data. takes the see and watch data 
def splitData(see, watch):
    # shuffles the data to randomize
    random.shuffle(see)
    random.shuffle(watch)
    # creates the splits for the data 
    seeSplit1, seeSplit2 = int(len(see) * .8), int(len(see) * .9)
    watchSplit1, watchSplit2 = int(len(watch) * .8), int(len(watch) * .9)
    # separates the data into train, dev, and test
    train = see[:seeSplit1] + watch[:watchSplit1]
    dev = see[seeSplit1:seeSplit2] + watch[watchSplit1:watchSplit2]
    test = see[seeSplit2:] + watch[watchSplit2:]
    # shuffles so its not just see then watch for all, though it does not matter
    random.shuffle(train)
    random.shuffle(dev)
    random.shuffle(test)
    return train, dev, test

"""creates and returns a context vector, made up of the left and right context vecctors concatonated together, 
for a given sentence, takes the data, the size of the vectors (so that they can match), and the vectors 
object so that each word's embedding can be searched for"""
def featurize(inData, vecSize, vectors):
    # holds the returnable data (after it has been featurized)
    outData = []
    for sentence, info, keyword in inData:
        sentence = sentence.split(' ')
        # creates context vectors for the left and right side
        rContext, lContext = np.zeros(shape=(1,vecSize)), np.zeros(shape=(1,vecSize))
        # true for left context, false for right context
        left = True 
        # counts the number of vectors added to each side
        rSummed, lSummed = 0, 0 
        # goes through each word in the sentence 
        for i, word in enumerate(sentence):
            # replaces w/ not so that we can retain negativity 
            if word == "n't":
                sentence[i] = 'not'
            # checks if the 'middle' of the sentence has been reached 
            if word == "*":
                left = False
            # builds left context vector 
            # .lower() is used as the glove embeddings do not factor in capitalization
            elif left == True:
                try:
                    lContext += vectors.get_vector(word.lower())
                    lSummed += 1
                except:
                    continue
            # builds right context vector
            elif left == False:
                try:
                    rContext += vectors.get_vector(word.lower()) 
                    rSummed += 1
                except:
                    continue
        # creates the normalized left and right contexts
        # also ensure no divide by 0 errors ie there is no context
        l = (lContext / lSummed)
        if np.isnan(l).any():
            l = np.zeros(shape=(1,vecSize))
        r = (rContext / rSummed)
        if np.isnan(r).any():
            r = np.zeros(shape=(1,vecSize))
        # concatonates the left and right contexts (both divided by the number of vectors added)
        context = np.concatenate((l, r), axis=None)
        # context vector, the sentence's info, and the keyword
        outData.append((context, info, keyword))
    return outData

#defines softmax function 
# ** change to logsoftmax **
def softmax(scores):
    exps = np.exp(scores)
    return exps / np.sum(exps)

# function classifies by returning the softmax of the  
def classify(featureVector, weightMatrix):
    scores = np.array([featureVector @ weightVector for weightVector in weightMatrix])
    return softmax(scores)

# function returns the accuracy of a trained weights matrix. takes the data, the weights, and the classes key
# (see and watch) so that they can be properly compared as the guess is numerical and the gold is a string
def accuracy(weightMatrix, featureMatrix, classes):
    # initializes the number of words that have been classified and the number of words correctly classified
    classified, correct = 0, 0
    # loops thorugh all of the given words
    for featureVector in featureMatrix:
        # increases the number of classified words
        classified += 1
        # determines the classification 
        probs = classify(featureVector[0], weightMatrix)
        index = max(probs)
        guess = np.where(probs == index)[0][0]
        # if the classification is correct add one to the correctly classified count
        if guess == classes[featureVector[2]]:
            correct += 1
    # after classifying all the words with the trained weights, returns the accuracy 
    return correct / classified

# function takes the training data, a weight matrix, a learning rate, the classes key, and the max number of 
# iterations   
def logisticRegression(trainData, weightMatrix, LR, classes, maxIts):
    print("Training...")
    # initializes variable to hold the number of iterations done 
    its = 0
    # goes until it reaches the max number of iterations
    for i in range(0,maxIts):
        print(f'Iteration {i+1}')
        # loops through every word breaking it up into the feature vector, the sentence info, and the gold
        for featureVector, info, gold in tqdm(trainData):
            # calculates the probabilities of each class for the given features
            probabilities = classify(featureVector, weightMatrix)
            # goes through all of the weights to adjust
            for i in classes:
                # determines if the current set of weights is the correct set (ie for a given class if it 
                # is the correct class) to set the y value
                y = 1.0 if i == gold else 0.0
                # changes the weights current class's weight values 
                weightMatrix[classes[i]] = weightMatrix[classes[i]] + LR * (y - 
                                                                probabilities[classes[i]]) * featureVector
        its += 1
    # returns the learned weights 
    return weightMatrix 

"""function reads in a file, finds the accuracy and creates a pandas dataframe so that each sentence can be 
investigated. the data frame is formated as the sentence, the human guess (using the most common guess 
from a survey of students), the computer guess, and the  computer's confidence in the guess. takes data, 
learned weights, and the word embeddings"""
def accAndDF(data, weights, vectors, vecSize):
    # classes key
    classes = {'see': 0, 'watch': 1}
    # the dictionary holding all the results the the sentence as the key
    results = {}
    # initialized num correct and classified
    correct, classified = 0, 0
    # goes through each line in the data and adds a number to be a place holder for the sentence data
    for line in data: line.insert(1, 1)
    # featurizes the data, goes through each example
    for test in featurize(data, vecSize, vectors):
        # classifies the example
        probs = classify(test[0], weights)
        index = max(probs)
        guess = np.where(probs == index)[0][0]
        # creates the string version of the guess
        guessWR = "see" if guess == 0 else "watch"
        # checks if the guess is correct
        if guess == classes[test[2]]:
            correct += 1
        # adds the results to the dict
        results[data[classified][0]] = [test[2], guessWR, index]
        classified += 1
    # prints the accuracy
    print(correct / classified)
    # returns a pandas dataframe of the results from the dict
    return pd.DataFrame.from_dict(results, orient="index", columns=["human", "computer", "comp certainty"])