from SuSyTruthChecker import SuSyTruthChecker

stc = SuSyTruthChecker("data/train.csv", "data/test.csv")

# In order of best accuracy:
dt_run = stc.train('dt')  # ~0.89
# gnb_run = stc.train('gnb')  # ~0.77
# knn_run = stc.train('knn')  # ~0.60

stc.perform_test(dt_run, fname_addition='dt')
