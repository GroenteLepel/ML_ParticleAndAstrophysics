from NetworkTrainer import NetworkTrainer
from tensorflow.keras import models, layers


class SusyCrossSectionPredictor(NetworkTrainer):
    def __init__(self, train_file: str, test_file: str, target: str = '',
                 standardised: bool = False, normalised: bool = False):
        super().__init__(train_file, test_file, target,
                         standardised, normalised)

    def train(self, method: str):
        if method == 'seq':
            model = models.Sequential()
            model.add(layers.Dense(64, 'relu',
                                   input_shape=(len(self.attributes),))
                      )
            model.add(layers.Dense(64, 'relu'))
            model.add(layers.Dense(64, 'relu'))
            model.add(layers.Dense(1, 'relu'))
            model.compile(optimizer='adam', loss='MSE')

            return self._train_model(model)
        else:
            raise Exception("No proper training method provided.")

    def _train_model(self, training_model):
        # Define for readability:
        t, att = self.target, self.attributes
        val_data = (self.val_data[att], self.val_data[t])

        history = training_model.fit(self.train_data[att],
                                     self.train_data[t],
                                     epochs=10,
                                     validation_data=val_data)

        return training_model
