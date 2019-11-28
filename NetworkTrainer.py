import time
from abc import abstractmethod

import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import store_results


class NetworkTrainer:

    def __init__(self, train_file: str, test_file: str, target: str = '',
                 standardised: bool = False,
                 normalised: bool = False
                 ):
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
        self.train_data, self.val_data = train_test_split(
            self.full_train_data, train_size=0.75)
        self.test_data = pandas.read_csv(test_file, header=0)

        # Set the attributes and target for the training data, which are all
        #  except the first (id) and final (truth_8tev) element.
        self.attributes = self.train_data.columns[1:-1]
        if target == '':
            self.target = self.train_data.columns[-1:][0]
        else:
            self.target = target

        self.is_normalised = normalised
        if standardised:
            self.standardise()
        self.is_standardised = standardised
        if normalised:
            self.normalise()

    def standardise(self):
        """
        Standardise all the data. This replaces the old data.
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

    @abstractmethod
    def _train_model(self, training_model):
        pass

    @abstractmethod
    def train(self, method: str):
        pass
