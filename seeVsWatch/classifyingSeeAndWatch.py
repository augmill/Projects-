#!/usr/bin/env python
# coding: utf-8

# This was originally written in jupyter using an anaconda environment

"""The goal of this model is to be able to guess whether the lemma see or watch takes the place of the target symbol * in a variety 
of contexts. The data used is from COCA and has been downloaded and can be worked with using a separate py file. There is an 
extra test set of made up sentences that are specifically about the viewing of digital media, this test set is the basis for 
a paper I wrote for my semantics class. """

# corpusmaker is the py file that holds my corpus class
from  corpusMaker import Corpus
import random
import gensim.downloader
import numpy as np
import pandas as pd
import gradio
from tqdm.autonotebook import tqdm
coca = Corpus()

# reads in all the files into the coca object 
files = ["wlp_acad.txt", "wlp_blog.txt", "wlp_fic.txt", "wlp_mag.txt", "wlp_news.txt", "wlp_spok.txt", "wlp_tvm.txt", "wlp_web.txt"]
coca.readInSeveral(files, "seeVsWatch/COCA/")

"""function goes through all the data and processes it by finding where in the setence the keyword is and 
replacing it with the target symbol * and adds a tuple of the processed sentence and the keyword. a set 
is used to prevent duplicate datum. takes the data"""
def processing(inData):
    # the data to be returned 
    outData = []
    # the set to prevent duplicates
    dataSet = set()
    for sentence in inData:
        #sentence split up
        splitSent = sentence[0].split(' ')
        # [the index is the keyword's index which is given as part of the kwic search]
        splitSent[sentence[1][3]] = '*'
        fullSent = ' '.join(splitSent)
        if (fullSent, sentence[2]) not in dataSet:
            dataSet.add((fullSent, sentence[2]))
            # sentence, sentence info, and sentence keyword
            outData.append((fullSent, sentence[1], sentence[2]))
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
    for data in inData:
        # holds the sentence's info [fileName, textNum, sentenceNum, wordNum]
        info = data[1]
        # holds the keyword (see or watch)
        keyword = data[2]
        # holds the split sentence data
        sentence = data[0].split(' ')
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
    for i in tqdm(range(0,maxIts)):
        # loops through every word breaking it up into the feature vector, the sentence info, and the gold
        for featureVector, info, gold in trainData:
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

# makes see and watch data
"""kwic is a typical keyword in context search, the one being done here is not case sensitive however, each
sentence also gets it's information of [fileName, textNum, sentenceNum, wordNum] where sentenceNum is 
specific to the text and wordNum is specific to the sentence. both are 0 indexed """
seeData = [sentence.append("see") or sentence  for sentence in coca.kwic("see")]
watchData = [sentence.append("watch") or sentence for sentence in coca.kwic("watch")]
# processes the data seperately 
see = processing(seeData)
watch = processing(watchData)
# splits the data into train, dev, and test data
train, dev, test = splitData(see, watch)

# sets the vectors 
vectors = gensim.downloader.load('glove-twitter-200')
# holds the vector 
vecSize = 200

# featurizes the data
featTrainData = featurize(train, vecSize, vectors)
featDevData = featurize(dev, vecSize, vectors)
featTestData = featurize(test, vecSize, vectors)

# creates the weights to be trained
weights = np.zeros(shape=(2,(vecSize * 2)))
# holds the classes for the 
classes = {'see': 0, 'watch': 1}
# learning rate and its can be adjusted
LR = 0.00035
its = 75
learnedWeights = logisticRegression(featTrainData, weights, LR, classes, its)

# after going through all the words checks the accuracy of the weights on the dev data
devAccuracy = accuracy(learnedWeights, featDevData, classes)
print("Dev accuracy:", devAccuracy)

# test set accuracy
accuracy(learnedWeights, featTestData, classes)

"""function reads in a file, finds the accuracy and creates a pandas dataframe so that each sentence can be 
investigated. the data frame is formated as the sentence, the human guess (using the most common guess 
from a survey of students), the computer guess, and the  computer's confidence in the guess. takes data, 
learned weights, and the word embeddings"""
def accAndDF(data, weights, vectors):
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
        results[lines[classified][0]] = [test[2], guessWR, index]
        classified += 1
    # prints the accuracy
    print(correct / classified)
    # returns a pandas dataframe of the results from the dict
    return pd.DataFrame.from_dict(results, orient="index", columns=["human", "computer", "comp certainty"])

# these create dataframes for each data set respectively
lines = [line[:-1].split('|') for line in open("seeVsWatch/test_data.txt")]
results = accAndDF(lines, learnedWeights, vectors)

lines = [line[:-1].split('|') for line in open("seeVsWatch/extra_tests.txt")]
extraResults = accAndDF(lines, learnedWeights, vectors)

lines = [line[:-1].split('|') for line in open("seeVsWatch/test_words.txt")]
wordResults = accAndDF(lines, learnedWeights, vectors)

"""# this is just to be able to create a gradio url that i can share

# function takes a sentence and makes it usable 
def takeText(text):
    # the 1 and gold are just place holders
    data = [[text, 1, "gold"],]
    return featurize(data, vecSize, vectors)
        
# takes a sentence and passes it to takeText and then classifies it returning the result
def classifier(text):
    featureVector = takeText(text)
    probs = classify(featureVector[0][0], learnedWeights)
    index = max(probs)
    guess = np.where(probs == index)[0][0]
    return "see" if guess == 0 else "watch"

# creates gradio share link
demo = gradio.Interface(fn=classifier, inputs='text', outputs='text')
demo.launch(share=True)"""

"""At this point my model has been trained and the investigation complete but it is worth checking it against
a scikit learn model to see how well I am doing, in comparison. I use a logistic regression model for
a direct comparison and a perceptron model just to see if my model does better than a different type. The
models will be checked using the COCA test data and the made up data, however confidence scores cannot be 
returned as the models just do binary classification not probabilities. So only their accuracies can be 
compared. """

# this makes the log regressions model 
from sklearn.linear_model import LogisticRegression
# this creates the features and class lists that the models need 
features = []
classes = []
for data in featTrainData:
    features.append(data[0])
    if data[2] == 'see':
        classes.append(0)
    else:
        classes.append(1)

# builds the log regression classifier
classifier = LogisticRegression(solver = 'lbfgs', multi_class = 'auto')
classifier.fit(features, classes)

# checks it using the test data
correctClasses = {'see': 0, 'watch': 1}
correct = 0
classified = 0

"""because the accuracy I had written requires different parameters than what the scikit models return
i just use a loop to run it quickly"""

for data in featTestData:
    if classifier.predict([data[0]]) == correctClasses[data[2]]:
        correct += 1
    classified += 1
correct / classified

"""function returns the accuracy of the scikit log regression model and returns a dataframe of the results
takes the data and word embeddings 
works same as the accAndDF function but using the scikit log regression predictions"""
def logRegAccAndDF(data, vectors):
    correctClasses = {'see': 0, 'watch': 1}
    correct = 0
    classified = 0
    results = {}
    for i, line in enumerate(data): line.insert(1, i)
    for test in featurize(data, vecSize, vectors):
        guess = classifier.predict([test[0]])
        guessWR = "see" if guess == 0 else "watch"
        if guess == correctClasses[test[2]]:
            correct += 1
        results[lines[classified][0]] = [guessWR]
        classified += 1
    print(correct / classified)
    return pd.DataFrame.from_dict(results, orient="index", columns=["log"])


lines = [line[:-1].split('|') for line in open("seeVsWatch/test_data.txt")]
logTests = logRegAccAndDF(lines, vectors)

lines = [line[:-1].split('|') for line in open("seeVsWatch/extra_tests.txt")]
logExtra = logRegAccAndDF(lines, vectors)

lines = [line[:-1].split('|') for line in open("seeVsWatch/test_words.txt")]
logWords = logRegAccAndDF(lines, vectors)


# this will allow me to test against the perceptron model
from sklearn.linear_model import Perceptron 
clf = Perceptron()
clf.fit(features, classes)

correctClasses = {'see': 0, 'watch': 1}
correct = 0
classified = 0
    
for data in featTestData:
    if clf.predict([data[0]]) == correctClasses[data[2]]:
        correct += 1
    classified += 1
correct / classified

"""function returns the accuracy of the scikit perceptron model and returns a dataframe of the results
takes the data and word embeddings 
works same as the accAndDF function but using the scikit perceptron predictions"""
def percepAccAndDF(data, vectors):
    correctClasses = {'see': 0, 'watch': 1}
    correct = 0
    classified = 0
    results = {}
    for i, line in enumerate(data): line.insert(1, i)
    for test in featurize(data, vecSize, vectors):
        guess = clf.predict([test[0]])
        guessWR = "see" if guess == 0 else "watch"
        if guess == correctClasses[test[2]]:
            correct += 1
        results[lines[classified][0]] = [guessWR]
        classified += 1
    print(correct / classified)
    return pd.DataFrame.from_dict(results, orient="index", columns=["percep"]) 


lines = [line[:-1].split('|') for line in open("seeVsWatch/test_data.txt")]
percepTests = percepAccAndDF(lines, vectors)


lines = [line[:-1].split('|') for line in open("seeVsWatch/extra_tests.txt")]
percepExtra = percepAccAndDF(lines, vectors)


lines = [line[:-1].split('|') for line in open("seeVsWatch/test_words.txt")]
percepWords = percepAccAndDF(lines, vectors)


"""Below are the dataframes for my model, the sckit log regression model, and the scikit perceptron models
for the test sentnces, the extra test sentences, and the test words. """
# test sentences
testFrames = [results, logTests, percepTests]
print("\nTest sentences:")
print(pd.concat(testFrames, axis=1))


# extra test sentences
extraFrames = [extraResults, logExtra, percepExtra]
print("\Extra sentences:")
print(pd.concat(extraFrames, axis=1))


# test words
wordFrames = [wordResults, logWords, percepWords]
print("\nTest words:")
print(pd.concat(wordFrames, axis=1))