from SuSyTruthChecker import SuSyTruthChecker

stc = SuSyTruthChecker("data/train.csv", "data/test.csv")

print(stc.train_data.size)
stc.train_knn()
stc._standardise()

print(stc.train_data.size)
stc.train_knn()