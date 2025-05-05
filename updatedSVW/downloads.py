from corpusMaker import Corpus
import gensim.downloader
from svwFuncs import *
# makes a corpus to hold the coca data
coca = Corpus()

# reads in all the files into the coca object 
coca.readInSeveral(["wlp_acad.txt", "wlp_blog.txt", "wlp_fic.txt", "wlp_mag.txt", 
    "wlp_news.txt", "wlp_spok.txt", "wlp_tvm.txt", "wlp_web.txt"], "/Users/augustmilliken/Documents/GitHub/Projects-/seeVsWatch/COCA/")

# sets the vectors 
vectors = gensim.downloader.load('glove-twitter-200')
# holds the vector 
vecSize = 200

# makes see and watch data
"""kwic is a typical keyword in context search, the one being done here is not case sensitive however, each
sentence also gets it's information of [fileName, textNum, sentenceNum, wordNum] where sentenceNum is 
specific to the text and wordNum is specific to the sentence. both are 0 indexed """
seeData = [sentence.append("see") or sentence  for sentence in coca.kwic("see", lemma=True)]
watchData = [sentence.append("watch") or sentence for sentence in coca.kwic("watch", lemma=True)]
# processes the data seperately 
see = processing(seeData, coca)
watch = processing(watchData, coca)
# splits the data into train, dev, and test data
train, dev, test = splitData(see, watch)

# featurizes the data
featTrainData = featurize(train, vecSize, vectors)
featDevData = featurize(dev, vecSize, vectors)
featTestData = featurize(test, vecSize, vectors)