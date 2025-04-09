#!/usr/bin/env python
# coding: utf-8

# This was originally written in jupyter using an anaconda environment
"""The goal of this model is to be able to guess whether the lemma see or watch takes the place of the target symbol * in a variety 
of contexts. The data used is from COCA and has been downloaded and can be worked with using a separate py file. There is an 
extra test set of made up sentences that are specifically about the viewing of digital media, this test set is the basis for 
a paper I wrote for my semantics class. """

# corpusmaker is the py file that holds my corpus class
from  corpusMaker import Corpus
from svwFuncs import *
from models import *
import gensim.downloader

# makes a corpus to hold the coca data
coca = Corpus()

# reads in all the files into the coca object 
coca.readInSeveral(["wlp_acad.txt", "wlp_blog.txt", "wlp_fic.txt", "wlp_mag.txt", "wlp_news.txt", "wlp_spok.txt", "wlp_tvm.txt", "wlp_web.txt"], "seeVsWatch/COCA/")

# makes see and watch data
"""kwic is a typical keyword in context search, the one being done here is not case sensitive however, each
sentence also gets it's information of [fileName, textNum, sentenceNum, wordNum] where sentenceNum is 
specific to the text and wordNum is specific to the sentence. both are 0 indexed """
seeData = [sentence.append("see") or sentence  for sentence in coca.kwic("see")]
watchData = [sentence.append("watch") or sentence for sentence in coca.kwic("watch")]
# processes the data seperately 
see = processing(seeData, coca)
watch = processing(watchData, coca)
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

# these create dataframes for each data set respectively
with open("seeVsWatch/test_data.txt") as file:
    results = accAndDF([line[:-1].split('|') for line in file], learnedWeights, vectors, vecSize)


with open("seeVsWatch/extra_tests.txt") as file:
    extraResults = accAndDF([line[:-1].split('|') for line in file], learnedWeights, vectors, vecSize)

with open("seeVsWatch/test_words.txt") as file:
    wordResults = accAndDF([line[:-1].split('|') for line in file], learnedWeights, vectors, vecSize)