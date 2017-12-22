import csv
import operator
from nltk import word_tokenize
from nltk import stem
import os
from dataManager import DataManager
from nltk.tag.stanford import CoreNLPPOSTagger, CoreNLPNERTagger
from nltk.tokenize.stanford import CoreNLPTokenizer
from nltk.tag.stanford import StanfordPOSTagger
import nltk
import pandas as pd
java_path = "C:\Program Files\Java\jdk1.8.0_151" # replace this
os.environ['JAVA_HOME'] = java_path
import re

class SubLexicon:
    def __init__(self, word, pos, isStem, polarity):
        self.__word = word
        self.__pos = self.get_pos(pos)
        self.__stemmed = \
            True if isStem == 'y' else False  # True if stemmed=y, else False
        self.__polarity = polarity

    def get_pos(self, pos):
        if pos == 'noun':
            return ['NN', 'NNS', 'NNP', 'NNPS']
        elif pos == 'adj':
            return ['JJ', 'JJR', 'JJS']
        elif pos == 'verb':
            return ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        else:
            return ['ANY']

    def get_polarity(self):
        return self.__polarity

class POS:
    nouns = []
    adjs = []
    verbs = []
    features_3_pos = []
    features_all_words = []
    subjectivity_lexicons = {}

    def extract_bow_3_pos_tags(self, filename):
        """
        2. a) Extract a bag-of-words list of nouns, adjectives, and verbs for all targets individually
        :param filename:
        :return:
        """
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                # print(row)
                if row[1] in ['NN', 'NNS', 'NNP']:
                    self.nouns.append(row[0])
                elif row[1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
                    self.verbs.append(row[0])
                elif row[1] in ['JJ', 'JJR', 'JJS']:
                    self.adjs.append(row[0])

        self.features_3_pos.extend(self.nouns)
        self.features_3_pos.extend(self.verbs)
        self.features_3_pos.extend(self.adjs)
        # remove duplicate features

        self.features_3_pos = list(set(self.features_3_pos))
        #print("features (after duplicates removed) are {}".format(self.features_3_pos.__len__()))

    def extract_bow_all_words(self, filename):
        """
        2. a) Extract a bag-of-words list of all words for all targets individually
        :param filename:
        :return:
        """
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                # print(row)
                self.features_all_words.append(row[0])

        # remove duplicate features
        self.features_all_words = list(set(self.features_all_words))
        #print("features (after duplicates removed) are {}".format(self.features_all_words.__len__()))

    def create_file(self, filename, new_filename, all_words=False):
        """
        Use those words as features and create a file in which the feature values are either 1 or 0 depending
        on whether the corresponding word is in the tweet or not. Add the tweet label as the last element
        (gold class) in every line.
        :param filename:
        :return:
        """
        with open(new_filename, 'w') as wp, open(filename, 'rU') as rp:
            writer = csv.writer(wp, delimiter=',')
            reader = csv.reader(rp, delimiter=',')
            # if not all_words:
            #     writer.writerow(self.features_3_pos)
            # else:
            #     writer.writerow(self.features_all_words)

            # f = open("test/trump_test.csv")
            # for row in f.readlines():
            #     print row
            n = 30
            for row in reader:
                #print row
                r = []
                nrow = []

                if len(row)>2:
                    nrow.append(row[:-2])
                    nrow.append(row[-2])
                    nrow.append(row[-1])

                    #text = word_tokenize(row[0])
                    pattern = r"'([A-Za-z0-9_\./\\-]*)'"
                    m = re.findall(pattern, str(nrow[0]))
                    nrow[0] = m
                    #print row[0], row[1], row[2]
                    if not all_words:
                        for feature in nrow[0]:
                            #print feature, row[0], feature in row[0]
                            if feature in self.features_3_pos:
                                r.append('1')
                            else:
                                r.append('0')
                    else:
                        for feature in nrow[0]:
                            if feature in self.features_all_words:
                                r.append('1')
                            else:
                                r.append('0')

                    rem = n - len(r)
                    #print n, len(r), rem
                    for l in range(rem):
                        r.append('0')
                    try:
                        #print r
                        #print nrow[-1], nrow
                        r = r[:n]
                        mylist = ["FAVOR", "AGAINST", "NONE"]
                        if(nrow[-1] in mylist):
                            r.append(nrow[-1])  # stance column
                            writer.writerow(r)
                            writer.write("\n")
                    except:
                        pass
    def read_subjectivity_lexicons(self, sublex_filename):
        """
        Reads the subjectivity lexicon file, and constructs the datastructure,
        finally adds to the 'subjectivity_lexicons' list
        :param sublex_filename:
        :return:
        """
        with open(sublex_filename) as rp:
            for row in rp:
                line_words = word_tokenize(row)
                # print line_words

                # parse the line
                lexicon_word = line_words[2].split('=')[1]
                lexicon_pos = line_words[3].split('=')[1]
                lexicon_stemmed = line_words[4].split('=')[1]
                lexicon_polarity = line_words[5].split('=')[1]

                # create new DS object & add to list
                self.subjectivity_lexicons[lexicon_word] = SubLexicon(word=lexicon_word,
                                                                      pos=lexicon_pos,
                                                                      isStem=lexicon_stemmed,
                                                                      polarity=lexicon_polarity)

        print ("total num. subjectivity lexicons = {}".format(self.subjectivity_lexicons.__len__()))
        print ("test lexicon polarity = {}".format(self.subjectivity_lexicons['abandoned'].get_polarity()))
        pass

    def create_features_with_sublex(self, filename, new_filename):
        with open(new_filename, 'w') as wp, open(filename, 'rU') as rp:
            writer = csv.writer(wp, delimiter=',')
            reader = csv.reader(rp, delimiter=',')

            # print (self.features_all_words)

            # create stemmer for extracting stems of words
            stemmer = stem.PorterStemmer()
            for row in reader:
                print  row
                r = []
                text = word_tokenize(row[0])


                for feature in self.features_all_words:
                    if feature in text:
                        # check if feature or the stem of the feature
                        #   is in subjectivity lexicon
                        feature_stem = stemmer.stem(feature)
                        feature = feature.lower()
                        if feature_stem in self.subjectivity_lexicons:
                            lexicon_obj = self.subjectivity_lexicons[feature_stem]
                            if lexicon_obj.get_polarity() == 'positive':
                                r.append('1')
                            else:
                                r.append('-1')
                        elif feature in self.subjectivity_lexicons:
                            lexicon_obj = self.subjectivity_lexicons[feature]
                            if lexicon_obj.get_polarity() == 'positive':
                                r.append('1')
                            else:
                                r.append('-1')
                        else:
                            r.append('0')
                    else:
                        r.append('0')

                r.append(row[2])  # stance column
                writer.writerow(r)

        pass
    #Tweet,Target,Stance,Opinion Towards,Sentiment

    def calculate_baseline(self, train_filename, test_filename):
        with open(train_filename, 'rU') as rp, open(test_filename, 'rU') as tp:
            reader_train = csv.reader(rp, delimiter=',')
            reader_test = csv.reader(tp, delimiter=',')

            # num. of examples in test
            test_data = list(reader_test)
            total_test_size = len(test_data)

            # count number of classes
            classes = {'FAVOR': 0, 'AGAINST': 0, 'NONE': 0}
            for row in reader_train:
                #print(row)

                classes[row[-1]] += 1 #row[2]

            max_class = max(classes.iteritems(), key=operator.itemgetter(1))[0]
            print ("max class for this data-set is: {}".format(max_class))

            misclassification_count = 0
            for row in test_data:
                if not row[-1] == max_class: #row[2]
                    misclassification_count += 1

            baseline_accuracy = float(total_test_size - misclassification_count) / float(total_test_size)
            print ("data-set size = {}, misclassification count = {}. Hence baseline accuracy = {}".format(total_test_size, misclassification_count, baseline_accuracy))
        pass

def makeDirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def pos_tag(series):
    # def rem_mentions_hasht(tweet):
    #     words = tweet.split()
    #     relevant_tokens = [w for w in words if '@' not in w and '#' not in w]
    #     return( " ".join(relevant_tokens))
    words  = list(series[0])
    #tokens = [w for w in words if '@' not in w and '#' not in w]
    stopSymbols = ["@", "#"]
    for word in range(len(words)):
        if words[word][0] in stopSymbols:
            #print words[word][0]
            words[word] = words[word][1:]
    #series = series.apply(lambda tweet: rem_mentions_hasht(tweet))
    #series = [rem_mentions_hasht(series[0]).split()]
    #print ("done 1")
    # from nltk.tag.stanford import StanfordPOSTagger
    # import os
    # java_path = "C:/Program Files/Java/jre1.8.0_111/bin/java.exe"
    # os.environ['JAVAHOME'] = java_path
    #coreNLPTokenizer = CoreNLPTokenizer()
    english_postagger = StanfordPOSTagger(java_options = "-mx7g",  model_filename='stanford_models/english-bidirectional-distsim.tagger', path_to_jar='stanford-postagger-3.8.0.jar')
    #from nltk.internals import find_jars_within_path
    #from nltk.parse.stanford import StanfordParser
    #english_postagger = StanfordParser(model_path="stanford_models/englishPCFG.ser.gz")
    #english_postagger._classpath = tuple(find_jars_within_path('.'))
    # #StanfordPOSTagger('stanford-postagger-3.8.0.jar')
    #print (nltk.pos_tag(words))
    x = english_postagger.tag(words)
    #print (x)
    return x
    #return series.apply(lambda a: english_postagger.tag(series))


if __name__ == '__main__':
    pos = POS()
    dp = DataManager('train.csv','test.csv')
    #print (dp.trainTweets," waah2")
    # train = pd.read_csv(open('train.csv', 'rU'))
    #test = pd.read_csv(open('test.csv', 'rU'))

    trainTargets = ["Hillary Clinton", "Legalization of Abortion", "Atheism", "Climate Change is a Real Concern", "Feminist Movement"]
    testTargets = ["Hillary Clinton", "Legalization of Abortion", "Atheism", "Climate Change is a Real Concern",
               "Feminist Movement", "Donald Trump"]
    trainFile = ["hillary", "abortion", "atheism", "climate", "feminist"]
    testFile = ["hillary", "abortion", "atheism", "climate", "feminist", "trump"]

    f1 = []
    for target in trainFile:
        f1.append(open("train/"+target+"_train.csv", "wb"))

    for tweet in dp.trainTweets:
        idx = trainTargets.index(tweet[1])
        words = list(tweet[0])
        stopSymbols = ["@", "#"]
        for word in range(len(words)):
            if words[word][0] in stopSymbols:
                words[word] = words[word][1:]
        #print ("f1 "+str(words) + "," + tweet[1] + "," + tweet[2])
        f1[idx].write(str(words) + "," + tweet[1] + "," + tweet[2])
        f1[idx].write("\n")

    g1 = []
    for target in testFile:
        g1.append(open("test/"+target+"_test.csv", "wb"))

    for tweet in dp.testTweets:
        idx = testTargets.index(tweet[1])
        words = list(tweet[0])
        stopSymbols = ["@", "#"]
        for word in range(len(words)):
            if words[word][0] in stopSymbols:
                words[word] = words[word][1:]
        #print ("g1 "+str(words) + "," + tweet[1] + "," + tweet[2])

        g1[idx].write(str(words)+ "," + tweet[1] + "," + tweet[2])
        g1[idx].write("\n")


    #print train.Tweet
    # train_stfrd_POStagged = []
    # k = 0

    # f = []
    # for target in trainFile:
    #     f.append(open("train/"+target.lower()+"_tagged_train.txt", "wb"))
    #
    # g = []
    # for target in testFile:
    #     g.append(open("test/"+target.lower()+"_tagged_test.txt", "wb"))


    # for tweet in dp.trainTweets:
    #     idx = trainTargets.index(tweet[1])
    #     #f[idx] = open("train/"+tweet[1]+"_pos_tagged.txt")
    #     # getAns = pos_tag(tweet)
    #     # test_stfrd_POStagged.append(getAns)
    #     x = pos_tag(tweet)
    #     for row in x:
    #         #print row[0], row[1], tweet[1], tweet[2]
    #         f[idx].write(str(row[0]+","+row[1]+","+tweet[1]+","+tweet[2]))
    #         f[idx].write("\n")
    #     k += 1
    #
    #     print(k)
    # print("done train POS")
    #
    # k = 0
    # for tweet in dp.testTweets:
    #     idx = testTargets.index(tweet[1])
    #     # f[idx] = open("train/"+tweet[1]+"_pos_tagged.txt")
    #     # getAns = pos_tag(tweet)
    #     # test_stfrd_POStagged.append(getAns)
    #     x = pos_tag(tweet)
    #     for row in x:
    #         # print row[0], row[1], tweet[1], tweet[2]
    #         g[idx].write(str(row[0] + "," + row[1] + "," + tweet[1] + "," + tweet[2]))
    #         g[idx].write("\n")
    #     k += 1
    #
    #     print(k)
    # print("done test POS")

    # test_stfrd_POStagged = []
    # k = 0
    # f = open("test/pos_tagged_test.txt", "wb")
    # for tweet in dp.testTweets:
    #     # getAns = pos_tag(tweet)
    #     # test_stfrd_POStagged.append(getAns)
    #     x = pos_tag(tweet)
    #     for row in x:
    #         #print row[0], row[1]
    #         f.write(str(row[0]+","+row[1]))
    #         f.write("\n")
    #     k += 1
    #     print(k)
    #
    # print("done test POS")
    # train_stfrd_POStagged = pos_tag(train.Tweet)
    # print("done test POS")

    #print (test_stfrd_POStagged)

    makeDirs("train")
    makeDirs("test")
    makeDirs("3_pos_tags")
    makeDirs("3_pos_tags/train")
    makeDirs("3_pos_tags/test")
    makeDirs("all_words")
    makeDirs("all_words/train")
    makeDirs("all_words/test")
    makeDirs("sublex_all_words")
    makeDirs("sublex_all_words/train")
    makeDirs("sublex_all_words/test")

    # pos_tagged_train = open("train/pos_tagged_train.txt", 'wb')
    # for tweet in dp.trainTweets:
    #     tweet = tweet[0]
    #     #words = tweet.split()
    #     # # relevant_tokens = [w for w in words if '@' not in w and '#' not in w]
    #     # # #return (" ".join(relevant_tokens))
    #     #
    #     # series = series.apply(lambda tweet: rem_mentions_hasht(tweet))
    #
    #     # java_path = "C:/Program Files/Java/jre1.8.0_111/bin/java.exe"
    #     # os.environ['JAVAHOME'] = java_path
    #     print(tweet)
    #     english_postagger = CoreNLPPOSTagger('stanford-postagger-3.8.0.jar')
    #
    #     #return tweet.apply(lambda a: english_postagger.tag(nltk.word_tokenize(a)))
    #     pos_tagged_train.write(tweet.apply(lambda a: english_postagger.tag(CoreNLPTokenizer.tokenize(a))))
    #

    for file in trainFile:
        pos.extract_bow_3_pos_tags("train/"+file+"_tagged_train.txt")
    for file in testFile:
        pos.extract_bow_3_pos_tags("test/"+file+"_tagged_test.txt")
    # pos.extract_bow_3_pos_tags("train/feminist_tagged_train.txt")
    # pos.extract_bow_3_pos_tags("test/feminist_tagged_test.txt")
    # print (len(POS.nouns))
    # print (len(POS.verbs))
    # print (len(POS.adjs))
    #print (len(POS.features_3_pos), "len")

    for file in trainFile:
        pos.create_file("train/"+file+"_train.csv",
                        "3_pos_tags/train/"+file+"_train_bow_features.csv")
    for file in testFile:
        pos.create_file("test/"+file+"_test.csv",
                        "3_pos_tags/test/"+file+"_test_bow_features.csv")

    # pos.extract_bow_all_words("train/hillary_tagged_train.txt")
    # pos.extract_bow_all_words("test/donald_tagged_test.txt")
    #
    # pos.create_file("hillary_train.csv",
    #                 "all_words/train/donald_train_bow_all_features.csv",
    #                 all_words=True)
    # pos.create_file("train_test_files/test/donald_test.csv",
    #                 "all_words/test/donald_test_bow_all_features.csv",
    #                 all_words=True)

    # pos.read_subjectivity_lexicons('subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff')
    # pos.create_features_with_sublex("feminist_train.csv",
    #                                 "sublex_all_words/train/feminist_train_sublex_all_features.csv")
    # pos.create_features_with_sublex("train_test_files/test/feminist_test.csv",
    #                                 "sublex_all_words/test/feminist_test_sublex_all_features.csv")

    for file in trainFile:
        print ("---"+file+"---")
        pos.calculate_baseline("3_pos_tags/train/" + file + "_train_bow_features.csv",
                                     "3_pos_tags/test/" + file + "_test_bow_features.csv")
        print ("--------------")
    #pos.calculate_baseline("train.csv", "test.csv")

