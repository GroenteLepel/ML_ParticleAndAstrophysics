from NetworkTrainer import NetworkTrainer


class SusyCrossSectionPredictor(NetworkTrainer):
    def __init__(self, train_file: str, test_file: str,
                 standardised: bool = False,
                 normalised: bool = False):
        super().__init__(train_file, test_file, 'cross_section',
                         standardised, normalised)
