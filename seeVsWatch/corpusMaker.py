# the class is one that allows for a corpus object that can hold texts and has, currently, building and a couple practical functions 
class Corpus:
    def __init__(self):
        # holds all the text numbers in the corpus
        self.textNums = []
        # holds all the data in the corpus format: {fileName: {textNumber: (wordData, sentences)}}
        self.data = {}
        # a key of what is in what position for the word data
        self.searchKey = {"word": 0, "lemma": 1, "pos": 2}
        self.textKey = {}
    
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
    
    # searchs for a variety of things depending on input
    def search(self, textNum, sentenceNum=None, wordNum=None, type=None, sentenceAsWords=False, fileName=None):
        # creates fileName if none given
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
        else:
            return self.data[fileName][textNum][0][sentenceNum][wordNum][self.searchKey[type]]

    # finds the file name for a given text 
    def findFileText(self, textNum):
        for file, texts in self.textKey.items():
            if textNum in texts:
                return file

    # returns the key word in context either from all files or just one text and will do so with or without case sensitivity
    def kwic(self, keyword, fileName=None, caseSensitive=False):
        sentences = []
        if caseSensitive == False:
            if fileName == None:
                for fileName, texts in self.data.items():
                    for text, textData in texts.items():
                        for i, sentence in enumerate(textData[1]):
                            words = sentence.lower().split()
                            if keyword.lower() in words:
                                sentences.append([sentence, [fileName, text, i, words.index(keyword)]])
            else: 
                for text, textData in self.data[fileName].items():
                    for i, sentence in enumerate(textData[1]):
                        words = sentence.lower().split()
                        if keyword.lower() in words:
                            sentences.append([sentence, [fileName, text, i, words.index(keyword)]])
        else:
            if fileName == None:
                for fileName, texts in self.data.items():
                    for text, textData in texts.items():
                        for i, sentence in enumerate(textData[1]):
                            words = sentence.split()
                            if keyword in words:
                                sentences.append([sentence, [fileName, text, i, words.index(keyword)]])
            else: 
                for text, textData in self.data[fileName].items():
                    for i, sentence in enumerate(textData[1]):
                        words = sentence.split()
                        if keyword in words:
                            sentences.append([sentence, [fileName, text, i, words.index(keyword)]])
        return sentences

    # returns a text or sentence as it's parts of speech either with or without the word
    def asPOS(self,textNum, sentenceNum=None, withWords=False):
        if sentenceNum == None:
            if withWords == False:
                return[word[2] for word in self.search(textNum, sentenceNum, sentenceAsWords=True)]
            else:
                return [(word[0], word[2]) for word in self.search(textNum, sentenceNum, sentenceAsWords=True)]
        else: 
            pos = []
            fileName = self.findFileText(textNum)
            if withWords == False:
                for i in range(len(self.data[fileName][textNum][0])):
                    pos.append([word[2] for word in self.data[fileName][textNum][0][1]])
            else:
                for i in range(len(self.data[fileName][textNum][0])):
                    pos.append([(word[0], word[2]) for word in self.data[fileName][textNum][0][1]])
            return pos
    

'''
to make: 
pos finder
sentence finder 
patern finder (pos with words)
key phrase in context
'''

