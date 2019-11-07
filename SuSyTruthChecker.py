import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas


class SuSyTruthChecker:
    """
    This class should read in a .csv file containing training data, train a
    model based on 19 free parameters to identify if a simulated signal should
    be true or not.
    """

    def __init__(self, train_file: str, test_file: str,
                 standardised: bool = False,
                 normalised: bool = False):
        """
        Read in the training and test data and store these in arrays. The
        structure of these files should be the following:
        ID, 20 parameters, truth value (only in training file)
        :param train_file: path to the file that contains parameters and labels.
        :param test_file: path to the file that contains only parameters.
        :param standardised: boolean indicating if the data needs to be
        standardised.
        """
        self.train_file = train_file
        self.test_file = test_file

        self.full_train_data = pandas.read_csv(train_file, header=0)
        # Since the test data does not have any labels, we cannot check how
        #  the trained algorithm will work on that set. For this we split the
        #  full training dataset into two parts: a train and check set.
        self.train_data, self.check_data = train_test_split(
            self.full_train_data, train_size=0.75)

        self.test_data = pandas.read_csv(test_file, header=0)

        # Set the attributes and target for the training data, which are all
        #  except the final element.
        self.attributes = self.train_data.columns[:-1]
        self.target = 'truth_8tev'

        if standardised:
            self._standardise()
        if normalised:
            self._normalise()

    def _standardise(self):
        # Define att for readability
        att = self.attributes
        standardised = \
            (self.train_data[att] - self.train_data[att].mean()) / \
            self.train_data[att].std()

        self.train_data.loc[:, att] = standardised
        print("This warning has been checked and can be ignored.\n")

    def _normalise(self):
        # Define att for readability
        att = self.attributes
        normalised = \
            (self.train_data[att] - self.train_data[att].min()) / \
            (self.train_data[att].max() - self.train_data[att].min())
        self.train_data.loc[:, att] = normalised
        print("This warning has been checked and can be ignored.\n")

    def train_knn(self, n_neighbours: int = 4):
        """
        Trains an algorithm based on the knn-method.
        :param n_neighbours: amount of neighbours that the method should use for
        indicating the data.
        :return: the accuracy of the trained set
        """
        start = time.time()
        clf = KNeighborsClassifier(n_neighbors=n_neighbours)
        clf.fit(self.train_data[self.attributes], self.train_data[self.target])

        prediction = clf.predict(self.check_data[self.attributes])
        accuracy = accuracy_score(self.check_data[self.target], prediction)

        print("Training time: %d seconds" % (time.time() - start))
        print(accuracy)
        return accuracy
