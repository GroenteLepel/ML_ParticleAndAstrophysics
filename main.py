from SuSyTruthChecker import SuSyTruthChecker

stc = SuSyTruthChecker("data/train.csv", "data/test.csv")

dt_run = stc.train('dt')
gnb_run = stc.train('gnb')

# This one is last since it normalises the data, and the other training methods
#  are influenced by this negatively.
knn_run = stc.train('knn')
