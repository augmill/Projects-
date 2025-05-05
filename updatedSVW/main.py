#!/usr/bin/env python
# coding: utf-8

# This was originally written in jupyter using an anaconda environment
"""The goal of this model is to be able to guess whether the lemma see or watch takes the place of the target symbol * in a variety 
of contexts. The data used is from COCA and has been downloaded and can be worked with using a separate py file. There is an 
extra test set of made up sentences that are specifically about the viewing of digital media, this test set is the basis for 
a paper I wrote for my semantics class. """

# corpusmaker is the py file that holds my corpus class
from downloads import *
from models import *

# creates the weights to be trained, vecSize * 2 accounts for concat context vec
# weights would have to increase to at least 7 for multi-class
# weights = np.zeros(shape=(2,(vecSize * 2)))
# holds the classes for the 
# classes = {'see': 0, 'watch': 1}
# classes = {''see' : ['see', 0], 'saw' : ['see', 1], 'seen' : ['see', 2], 'sees' : ['see', 3], 'watch' : ['watch', 4], 
# 'watched' : ['watch', 5], 'watches' : ['watch', 6]}
# learning rate and its can be adjusted
# classes = {'see': 0, 'saw' : 1, 'seen' : 2, 'sees': 3, 'watch' : 4, 'watched' : 5, 'watches' : 6}
# 00035
# 20 = 
'''
weights = np.zeros(shape=(7,(vecSize * 2)))
classes = {'see': 0, 'saw' : 1, 'seen' : 2, 'sees': 3, 'watch' : 4, 'watched' : 5, 'watches' : 6}
LR = 0.002
its = 75
learnedWeights = logisticRegression(featTrainData, weights, LR, classes, its)
print(f'Dev acc: {accuracy(learnedWeights, featDevData, classes)}')
'''

# after going through all the words checks the accuracy of the weights on the dev data
# devAccuracy = accuracy(learnedWeights, featDevData, classes)
# print("Dev accuracy:", devAccuracy)

# test set accuracy
# print(f'Test acc: {accuracy(learnedWeights, featTestData, classes)}')

# these create dataframes for each data set respectively
# with open("/Users/augustmilliken/Documents/GitHub/Projects-/seeVsWatch/test_data.txt") as file:
#     results = accAndDF([line[:-1].split('|') for line in file], learnedWeights, vectors, vecSize, classes)


# with open("/Users/augustmilliken/Documents/GitHub/Projects-/seeVsWatch/extra_tests.txt") as file:
#     extraResults = accAndDF([line[:-1].split('|') for line in file], learnedWeights, vectors, vecSize, classes)

# with open("/Users/augustmilliken/Documents/GitHub/Projects-/seeVsWatch/test_words.txt") as file:
#     wordResults = accAndDF([line[:-1].split('|') for line in file], learnedWeights, vectors, vecSize, classes)