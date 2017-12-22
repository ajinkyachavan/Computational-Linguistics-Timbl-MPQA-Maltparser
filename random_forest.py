from sklearn.ensemble import RandomForestClassifier
import numpy as np

class RandomForest:

    def __init__(self, train_file, test_file):
        self.__train_file = train_file
        self.__test_file = test_file
        self.__train_data = []
        self.__train_labels = []
        self.__test_data = []
        self.__test_labels = []
        self.mylist = ["FAVOR","NONE", "AGAINST"]

    def read_data(self):
        with open(self.__train_file, 'rU') as rp:
            for row in rp:
                train_data = row.split(',')
                idx = int(self.mylist.index(str(train_data[-1][:-1])))
                train_label = idx
                train_data = train_data[:-1]
                #print train_data


                # for i in range(len(train_data)):
                #    train_data[i] = list(map(int, train_data[i]))
                #new_train_data = [[] * len(train_data[0])] *len(train_data)
                #print "size tr",len(train_data[0]), len(train_data)

                # print ("train data = {}, \n\n train_label = {}".format(train_data, train_label))
                # for i in range(len(train_data)):
                #     for j in range(len(train_data[i])):
                #         new_train_data[i][j] = int(train_data[i][j])
                self.__train_data.append(train_data)
                self.__train_labels.append(train_label)
            for i in range(len(self.__train_data)):
                self.__train_data[i] = list(map(int, self.__train_data[i]))
                #print len(self.__train_data[i])
                self.__train_data[i] = self.__train_data[i][:30]
            #print len(self.__train_data), len(self.__train_data[0]), len(self.__train_labels), "lab"

            print ("num. train examples = {}, num. train labels = {}".format(self.__train_data.__len__(),
                                                                             self.__train_labels.__len__()))

        with open(self.__test_file, 'rU') as rp:
            for row in rp:
                test_data = row.split(',')
                idx = int(self.mylist.index(str(test_data[-1])[:-1]))
                test_label = idx
                test_data= test_data[:-1]


                #new_test_data = [[] * len(test_data)] *len(test_data[0])
                # print ("test data = {}, \n\n test_label = {}".format(test_data, test_label))
                # for i in range(len(test_data)):
                #     for j in range(len(test_data[i])):
                #         new_test_data[i][j] = int(test_data[i][j])
                self.__test_data.append(test_data)
                self.__test_labels.append(test_label)
            for i in range(len(self.__test_data)):
                self.__test_data[i] = list(map(int, self.__test_data[i]))
                self.__test_data[i] = self.__test_data[i][:30]

            #print len(self.__test_data), len(self.__test_data[0]), len(self.__test_labels), "lab"

            print ("num. test examples = {}, num. test labels = {}".format(self.__test_data.__len__(),
                                                                             self.__test_labels.__len__()))
            #print self.__train_data
            #print self.__train_labels

    def learn(self):
        forest = RandomForestClassifier(n_estimators=100)
        # mytrainarr = [[]*30]*543
        # for row in self.__train_data:
        #     mytrainarr.append(row)
        # mytestarr = [[]*30]*285
        # for row in self.__test_data:
        #     mytestarr.append(row)

        self.__train_data = np.reshape(self.__train_data, (len(self.__train_data),30))
        self.__test_data = np.reshape(self.__test_data, (len(self.__test_data),30))
        #print (self.__test_data, "idhar")
        forest = forest.fit(self.__train_data, self.__train_labels)
        #test_data = [[4,5]]
        #forest = forest.fit([[1,2],[3,4]], [3,7])
        #print forest
        self.__predicted_labels = forest.predict(self.__test_data)

        print (len(self.__predicted_labels), len(self.__test_labels))
    def calculate_accuracy(self):

        misclassification_count = 0
        for index, predicted_label in enumerate(self.__predicted_labels):
            if(not self.__test_labels[index] == predicted_label):
                misclassification_count += 1

        accuracy = float(self.__test_labels.__len__() - misclassification_count) / float(self.__test_labels.__len__())

        print ("Random Forest Accuracy = {}".format(accuracy))
        pass

if __name__ == "__main__":

    trainFile = ["hillary", "abortion", "atheism", "climate", "feminist"]

    for file in trainFile:
        random_forest = RandomForest("3_pos_tags/train/"+file+"_train_bow_features.csv",
                                     "3_pos_tags/test/"+file+"_test_bow_features.csv")
        print ("---"+file+"----")
        random_forest.read_data()
        random_forest.learn()
        random_forest.calculate_accuracy()
        print ("-----------\n")