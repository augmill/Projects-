# the class is one that allows for a corpus object that can hold texts and has, currently, a building and a couple, practical functions 
class Corpus:
    def __init__(self):
        # holds all the text numbers in the corpus
        self.textNums = []
        # holds all the data in the corpus format: {fileName: {textNumber: (wordData, sentences)}}
        self.data = {}
        # a key of what is in what position for the word data
        self.searchKey = {"word": 0, "lemma": 1, "pos": 2}
        self.textKey = {}
        '''# holds the file type so that the correct handling occurs
        self.fileType = ""'''
    
    # reads in a file to the corpus
    def readInFile(self, file, path=None):
        fileName = file[4:-4]
        # reads in the file line by line
        if path == None: lines = [line[:-1].split('\t') for line in open(file)]
        else: lines = [line[:-1].split('\t') for line in open(path+file)]
        # holds all the texts and their words format {textNumber: (wordData, sentences)}
        data = {}
        # holds all of the text numbers from that file
        fileTexts = {fileName: []}
        # holds the given text number
        textNum = 0
        # holds all of words in a given text
        textWordData = []
        # holds all the word data for a given sentence (added to textWordData)
        sentenceWordData = []
        # holds all the sentences of a text 
        textSentences = []
        # holds each sentence (added to textSentences) 
        sentence = ""
        # loops through each line of the file
        for i, word in enumerate(lines):
            # checks if it is the start of a new text and if so adds the text number to the list of them and then skips to the next line
            if word[1][:2] == '@@':
                textNum = int(word[0])
                self.textNums.append(textNum)
                continue
            # adds the word to the current sentence 
            sentence += word[1] + " " 
            # adds the word to the sentence data but skips the text number
            sentenceWordData.append(word[1:])
            # checks if it is the end of a sentence and if so adds the sentence and the sentence data to the text's sentences and text's
            # sentence data respectively
            if word[1] in ['.', '!', '?'] and (word[3] in ['y', '#y', '.', '!', '?']):
                textSentences.append(sentence)
                sentence = ""
                textWordData.append(sentenceWordData)
                sentenceWordData = []
            # checks if it is the end of the text and if so adds the text's data and sentences to its dictionary value as a tuple
            if lines[i+1][0] == 'END' or lines[i+1][1][:2] == '@@':
                data[textNum] = (textWordData, textSentences)
                fileTexts[fileName].append(textNum)
                textWordData = []
                textSentences = []
                # checks if it is the end of the file
                if lines[i+1][0] == 'END':
                    break
        self.data[fileName] = data
        self.textKey.update(fileTexts) 
     
    # reads in a list of files and provides space for a path as necessary
    def readInSeveral(self, files, path=None):
        for file in files:
            if path == None: self.readInFile(file, path)
            else: self.readInFile(file, path)
    
    # searchs for a variety of things depending on input, requires at least a text number 
    def search(self, textNum, sentenceNum=None, wordNum=None, type=None, sentenceAsWords=False, fileName=None):
        # finds the relevant fileName if none given
        if fileName == None:
            fileName = self.findFileText(textNum)
        # returns a search for the sentences of a given text
        if sentenceNum == None:
            return self.data[fileName][textNum][1]
        # returns a search for a sentence from a given text as a regular sentence
        elif wordNum == None and sentenceAsWords == False:
            return self.data[fileName][textNum][1][sentenceNum]
        # returns a search for a sentence from a given text as word by word data
        elif wordNum == None and sentenceAsWords == True:
            return self.data[fileName][textNum][0][sentenceNum]
        # returns a search for word data from a given sentence in a given text 
        elif type == None:
            return "Word: {}, Lemma: {}, POS: {}".format(*self.data[fileName][textNum][0][sentenceNum][wordNum])
        # returns a search for the specific word info from a word from a given sentence from a given text
        # type is word using 0, lemma using 1, or pos using 2
        else:
            return self.data[fileName][textNum][0][sentenceNum][wordNum][self.searchKey[type]]

    # finds the file name for a given text 
    def findFileText(self, textNum):
        for file, texts in self.textKey.items():
            if textNum in texts:
                return file

    # returns all sentences with the keyword in context either from all files or just one text and will do so with or without case sensitivity
    def kwic(self, keyword, fileName=None, caseSensitive=False):
        # creates a list to hold all sentences with the target sentence, with given parameters
        sentences = []
        # if the search is not case sensetive
        if caseSensitive == False:
            # if the search is for a the whole corpus
            if fileName == None:
                # loops through every sentence in corpus 
                for fileName, texts in self.data.items():
                    for text, textData in texts.items():
                        for i, sentence in enumerate(textData[1]):
                            # breaks the sentence into the individual words 
                            words = sentence.lower().split()
                            # checks if the keyword (ignroing case) is in the sentence and if so adds the sentence to the list of target sentences 
                            if keyword.lower() in words:
                                sentences.append([sentence, [fileName, text, i, words.index(keyword)]])
            # if the search is for a specific file
            else: 
                # loops through each sentence in the given file
                for text, textData in self.data[fileName].items():
                    for i, sentence in enumerate(textData[1]):
                        # breaks the sentence into the individual words
                        words = sentence.lower().split()
                        # checks if the keyword (ignroing case) is in the sentence and if so adds the sentence to the list of target sentences
                        if keyword.lower() in words:
                            sentences.append([sentence, [fileName, text, i, words.index(keyword)]])
        # if the search is case sensitive
        else:
            # if the search is for a the whole corpus
            if fileName == None:
                #loops through every sentence in corpus
                for fileName, texts in self.data.items():
                    for text, textData in texts.items():
                        for i, sentence in enumerate(textData[1]):
                            # breaks the sentence into the individual words
                            words = sentence.split()
                            # checks if the keyword is in the sentence and if so adds the sentence to the list of target sentences
                            if keyword in words:
                                sentences.append([sentence, [fileName, text, i, words.index(keyword)]])
            # if there is a file name
            else: 
                # if the search is for a specific file
                for text, textData in self.data[fileName].items():
                    for i, sentence in enumerate(textData[1]):
                        # breaks the sentence into the individual words
                        words = sentence.split()
                        if keyword in words:
                            # checks if the keyword is in the sentence and if so adds the sentence to the list of target sentences
                            sentences.append([sentence, [fileName, text, i, words.index(keyword)]])
        return sentences

    # returns a text or sentence as it's parts of speech either with or without the words
    def asPOS(self, textNum, sentenceNum=None, withWords=False):
        # if parts of speech are desired for whole text
        if sentenceNum == None:
            # sets the file name given the text number 
            fileName = self.findFileText(textNum)
            # if the search is for only the part of speech 
            if withWords == False:
                return [[word[2] for word in sent] for sent in self.data[fileName][textNum][0]]
            # if the searcg us for the word and its part of speech
            else:
                return [[(word[0], word[2]) for word in sent] for sent in self.data[fileName][textNum][0]]
        # if the parts of speech are desired for a specific sentence
        else: 
            # if the search is for only the part of speech 
            if withWords == False:
                return[word[2] for word in self.search(textNum, sentenceNum, sentenceAsWords=True)]
            # if the search is for the word and its part of speech
            else:
                return [(word[0], word[2]) for word in self.search(textNum, sentenceNum, sentenceAsWords=True)]
            
    

'''
These are extra function I would like to add
to make: 
pos finder
sentence finder 
patern finder (pos with words)
key phrase in context
'''

