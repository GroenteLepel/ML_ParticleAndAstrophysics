import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas

class SuSyTruthCecker:
    """
    This class should read in a .csv file containing training data, train a
    model based on 19 free parameters to identify if a simulated signal should
    be true or not.
    """

    def __init__(self, train_file: str, test_file: str):
        """
        Read in the training and test data and store these in arrays. The
        structure of these files should be the following:
        ID, 20 parameters, truthvalue (only in training file)
        :param train_file:
        :param test_file:
        """
        self.train_file = train_file
        self.test_file = test_file

        self.train_data = pandas.read_csv(train_file, header=0)
        self.test_data = pandas.read_csv(test_file, header=0)

    def train(self):
        """

        :return:
        """
        attributes = self.train_data.columns[:-1]
        target = 'truth_8tev'

        train, test = train_test_split(self.train_data, train_size=0.75)

        clf = KNeighborsClassifier(n_neighbors=4)
        clf.fit(train[attributes], train[target])

        prediction = clf.predict(test[attributes])
        accuracy = accuracy_score(test[target], prediction)
        print(accuracy)
