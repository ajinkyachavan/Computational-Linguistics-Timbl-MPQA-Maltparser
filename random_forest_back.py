from sklearn.ensemble import RandomForestClassifier
import numpy as np

class RandomForest:

    def __init__(self, train_file, test_file):
        self.train_file = train_file
        self.test_file = test_file
        self.train_data = []
        self.train_labels = []
        self.test_data = []
        self.test_labels = []
        self.mylist = ["FAVOR","NONE", "AGAINST"]
    def read_data(self):
        with open(self.train_file, 'rU') as rp:
            for row in rp:
                train_data = row.split(',')
                print train_data
                idx = int(self.mylist.index(str(train_data[-1][:-1])))
                train_label = idx
                train_data = train_data[:-1]
                for data in range(len(train_data)):
                    train_data[data] = int(train_data[data])
                #self.train_data = np.array(self.train_data)
                #self.train_label = np.array(self.train_label)
                #print ("train data = {}, \n\n train_label = {}".format(train_data, train_label))

                self.train_data.extend(train_data)
                self.train_labels.extend(train_label)

                print (self.train_data)

            print ("num. train examples = {}, num. train labels = {}".format(len(self.train_data),
                                                                             len(self.train_labels)))

        with open(self.test_file, 'rU') as rp:
            for row in rp:
                test_data = row.split(',')
                idx = int(self.mylist.index(str(test_data[-1])[:-1]))
                test_label = idx
                #test_label = test_data[test_data.len() - 1]
                test_data = test_data[:-1]
                
                for data in range(len(test_data)):
                    test_data[data] = int(test_data[data])
                #self.test_data = np.array(self.test_data)
                #print ("test data = {}, \n\n test_label = {}".format(test_data, test_label))

                self.test_data.extend(test_data)
                self.test_labels.extend(test_label)

            print ("num. test examples = {}, num. test labels = {}".format(len(self.test_data),
                                                                             len(self.test_labels)))


    def learn(self):
        forest = RandomForestClassifier(n_estimators=100)
        forest = forest.fit(self.train_data, self.train_labels)

        self.predicted_labels = forest.predict(self.test_data)


    def calculate_accuracy(self):

        misclassification_count = 0
        for index, predicted_label in enumerate(self.predicted_labels):
            if(not self.test_labels[index] == predicted_label):
                misclassification_count += 1

        accuracy = float(self.test_labels.len() - misclassification_count) / float(self.test_labels.len())

        print ("Random Forest Accuracy = {}".format(accuracy))
        pass

if __name__ == "__main__":
    random_forest = RandomForest("3_pos_tags/train/hillary_train_bow_features.csv",
                                 "3_pos_tags/test/hillary_test_bow_features.csv")

    random_forest.read_data()
    random_forest.learn()
    random_forest.calculate_accuracy()