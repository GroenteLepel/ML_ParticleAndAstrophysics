from NetworkTrainer import NetworkTrainer

import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier


class SuSyTruthChecker(NetworkTrainer):
    """
    This class should read in a .csv file containing training data, train a
    model based on 19 free parameters to identify if a simulated signal should
    be true or not.
    """

    def __init__(self, train_file: str, test_file: str, target: str = '',
                 standardised: bool = False, normalised: bool = False):
        super().__init__(train_file, test_file, target,
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
            raise Exception("No proper training method provided.")
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
