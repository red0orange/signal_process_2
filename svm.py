import numpy as np
import sklearn.svm
from sklearn.metrics import precision_score, recall_score


if __name__ == "__main__":
    train_data_path = "/home/dehao/github_projects/signal_process_2/ml_data/train.npy"
    test_data_path = "/home/dehao/github_projects/signal_process_2/ml_data/test.npy"

    train_data = np.load(train_data_path)
    test_data = np.load(test_data_path)

    train_X, train_Y = train_data[:, 1:], train_data[:, 0]
    test_X, test_Y = test_data[:, 1:], test_data[:, 0]

    svm = sklearn.svm.SVC(C = 300000.0, kernel="rbf")
    svm.fit(train_X, train_Y)
    predict_Y = svm.predict(test_X)

    print(np.sum(predict_Y == test_Y) / len(test_Y))
    print(precision_score(test_Y, predict_Y), recall_score(test_Y, predict_Y))
    test_Y, predict_Y = [not i for i in test_Y], [not i for i in predict_Y]
    print(precision_score(test_Y, predict_Y), recall_score(test_Y, predict_Y))
    pass