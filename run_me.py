from SusyCrossSectionPredictor import SusyCrossSectionPredictor

scsp = SusyCrossSectionPredictor('data/cross_sections_train.csv',
                                 'data/cross_sections_test.csv')
scsp.train('seq')
