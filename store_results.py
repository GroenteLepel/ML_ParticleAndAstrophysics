"""
# Store Results

The store function in this file stores the prediction results in the format needed for scoring by the scorings website for the course 'Machine Learning in Particle Physics and Astronomy'. It has four input arguments:

	- ids: list or numpy array containing the IDs of the data points.
	- predictions: list or numpy array containing the predictions for the data points. The number of predictions and provided IDs should match and the orders should correspond to eachother (i.e. the first element of the predictions array should have as its ID the first element of the provided ids list or array).
	- save_location: If set to a location, the results will be stored in said location. This argument is optional. If it is set to None (default) the result will not be stored by this function.
	- print_results: If set to True (default) the output will be printed to the screen or terminal. If set to False, the result will not be printed.

If we assume the ID is the first column in our data set, the target variable (the thing we want to predict) the last one and every variable in between is used for making our prediction, we can use the following code to train a decision tree and output the result:

	from store_results import store
	from sklearn.tree import DecisionTreeClassifier
	import pandas as pd

	train = pd.read_csv("train.csv")
	test = pd.read_csv("test.csv")
	labels = train.keys()[1:-1]
	target = train.keys()[-1]

	clf = DecisionTreeClassifier()
	clf.fit(train[labels], train[target])
	ypred = clf.predict(test[labels])

	result = store( test["id"], ypred )


"""

import pandas as pd
import numpy as np


def store(ids, predictions, save_location=None, print_results=True):
    if len(ids) != len(predictions):
        raise Exception(
            "Number of provided IDs should match the number of provided predictions")
    if isinstance(ids, pd.core.series.Series):
        ids = ids.values
    if not isinstance(ids, np.ndarray) and not isinstance(ids, list):
        raise Exception(
            "IDs should be provided as a numpy array, pandas Series (column in a pandas data frame) or a list. Provided was a {}".format(
                type(ids)))
    if not isinstance(predictions, np.ndarray) and not isinstance(ids, list):
        raise Exception(
            "IDs should be provided as a numpy array or a list. Provided was a {}".format(
                type(predictions)))

    output = "id,prediction\n"
    for i, p in zip(ids, predictions):
        output += "{},{}\n".format(i, p)
    if save_location != None:
        with open(save_location, "w") as f:
            f.write(output)
    if print_results:
        print(output)
    return output
