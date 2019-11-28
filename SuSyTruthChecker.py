import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import pandas

import store_results
from NetworkTrainer import NetworkTrainer


class SuSyTruthChecker(NetworkTrainer):
    """
    This class should read in a .csv file containing training data, train a
    model based on 19 free parameters to identify if a simulated signal should
    be true or not.
    """

    def __init__(self, train_file: str, test_file: str,
                 standardised: bool = False,
                 normalised: bool = False):
        super().__init__(train_file, test_file, 'truth_8tev',
                         standardised, normalised)
        self.method = 'none'

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
            self.method = 'k nearest neighbours'
            model = KNeighborsClassifier(n_neighbors=5)
            return self._train_model(model)
        elif method == 'dt':
            # Best result came from doing nothing with the data, and resulted
            #  in an accuracy of ~0.89.
            self.method = 'decision tree'
            model = DecisionTreeClassifier(
                min_samples_leaf=4,
                min_samples_split=10
            )
            return self._train_model(model)
        elif method == 'gnb':
            # This method is based on the Bayesian probability that a point in
            #  the data set is a certain class, e.g. p(x = 1), given all the
            #  parameters for this point, y_i, so e.g. p(x = 1 | y_i). The naive
            #  part of the method is that it considers that all these parameters
            #  y_i are independent of each other.
            # This method was just implemented to see the documentation from
            #  scikit-learn, no real experimenting has been done. This delivered
            #  an accuracy of ~0.78.
            self.method = 'naive bayes (gaussian)'
            model = GaussianNB()
            return self._train_model(model)
        elif method == 'adaboost':
            self.method = method
            model = AdaBoostClassifier(n_estimators=10)
            return self._train_ensemble(model)
        else:
            print("No proper training method given.")
            return 0

    def _train_model(self, training_model):
        start = time.time()
        training_model.fit(self.train_data[self.attributes],
                           self.train_data[self.target])

        prediction = training_model.predict(self.val_data[self.attributes])
        accuracy = accuracy_score(self.val_data[self.target], prediction)

        print("Training time: %d seconds" % (time.time() - start))
        print("Accuracy of model:", accuracy)

        return training_model

    def _train_ensemble(self, training_model):
        start = time.time()
        scores = cross_val_score(training_model,
                                 self.full_train_data[self.attributes],
                                 self.full_train_data[self.target],
                                 cv=5)

        print("Training time: %d seconds" % (time.time() - start))
        print("Accuracy of model:", scores.mean())
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
