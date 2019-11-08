import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas

import store_results


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
        :param normalised: boolean indicating if the data needs to be
        normalised.
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
        #  except the first (id) and final (truth_8tev) element.
        self.attributes = self.train_data.columns[1:-1]
        self.target = 'truth_8tev'

        self.is_normalised = normalised
        if standardised:
            self.standardise()
        self.is_standardised = standardised
        if normalised:
            self.normalise()

    def standardise(self):
        """
        Standardise all the data. This replaces the old data.
        :return:
        """
        print("Standardising data.")
        # Define att for readability
        att = self.attributes
        standardised = \
            (self.train_data[att] - self.train_data[att].mean()) / \
            self.train_data[att].std()

        self.train_data.loc[:, att] = standardised
        print("This warning has been checked and can be ignored.\n")

        self.is_standardised = True

    def normalise(self):
        """
        Normalise all the data. This replaces the old data.
        :return:
        """
        print("Normalising data.")
        # Define att for readability
        att = self.attributes
        normalised = \
            (self.train_data[att] - self.train_data[att].min()) / \
            (self.train_data[att].max() - self.train_data[att].min())
        self.train_data.loc[:, att] = normalised
        print("This warning has been checked and can be ignored.\n")

        self.is_normalised = True

    def train(self, method: str):
        """
        Method for selecting a preconfigured training method applied to the
        dataset.
        :param method: ['knn', 'dt', 'gnb'].
        :return:
        """
        if method == 'knn':
            # 'Best' result came from had an accuracy of ~0.59, which is just a
            #  bit better than guessing.
            method = 'k nearest neighbours'
            model = KNeighborsClassifier(n_neighbors=5)
        elif method == 'dt':
            # Best result came from doing nothing with the data, and resulted
            #  in an accuracy of ~0.89.
            method = 'decision tree'
            model = DecisionTreeClassifier(
                min_samples_leaf=4,
                min_samples_split=10
            )
        elif method == 'gnb':
            # This method is based on the Bayesian probability that a point in
            #  the data set is a certain class, e.g. p(x = 1), given all the
            #  parameters for this point, y_i, so e.g. p(x = 1 | y_i). The naive
            #  part of the method is that it considers that all these parameters
            #  y_i are independent of each other.
            # This method was just implemented to see the documentation from
            #  scikit-learn, no real experimenting has been done. This delivered
            #  an accuracy of ~0.78.
            method = 'naive bayes (gaussian)'
            model = GaussianNB()
        else:
            print("No proper training method given.")
            return 0

        print("Training data based on model", method, ".")
        return self._train_model(model)

    def _train_model(self, training_model):
        start = time.time()
        training_model.fit(self.train_data[self.attributes],
                           self.train_data[self.target])

        prediction = training_model.predict(self.check_data[self.attributes])
        accuracy = accuracy_score(self.check_data[self.target], prediction)

        print("Training time: %d seconds" % (time.time() - start))
        print("Accuracy of model:", accuracy)

        return training_model

    def perform_test(self, trained_model, fname_addition: str = ''):
        """
        Applies the trained model to the test data, and stores the results in
        a .csv file with the help of the provided store_results.py file.
        :param fname_addition: addition to the filename at which the data is
        stored for better recognition.
        :param trained_model: model trained by scikit-learn which must be
        applied to the test data.
        :return:
        """
        if fname_addition != '':
            fname = 'data/' + fname_addition + '_res.csv'
        else:
            fname = 'data/res.csv'
        prediction = trained_model.predict(self.test_data[self.attributes])
        store_results.store(self.test_data['id'], prediction,
                            save_location=fname)
