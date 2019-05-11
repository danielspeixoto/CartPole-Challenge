import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import metrics
import pandas as pd
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support as score

def k_fold(k, df, y, clf):
    fold = StratifiedKFold(k)
    final_accuracy = 0.0
    final_precision = 0.0
    final_recall = 0.0
    final_fscore = 0.0
    for train, test in fold.split(df, y):
        X_train, X_test = list(pd.Series(df)[train]), list(pd.Series(df)[test])
        y_train, y_test = list(pd.Series(y)[train]), list(pd.Series(y)[test])

        train = []
        test = []
        for t in y_train:
            train.append(int(t))

        for t in y_test:
            test.append(int(t))



        clf = clf.fit(X_train, train)
        f_predicted = np.round(np.clip(clf.predict(X_test), 0, 1))

        predicted = []
        for p in f_predicted:
            predicted.append(int(p))

        accuracy = 0

        for i in range(len(predicted)):
            if predicted[i] == test[i]:
                accuracy += 1
        accuracy = accuracy / len(predicted)

        final_accuracy = final_accuracy + accuracy

        precision, recall, fscore, _ = score(test, predicted,
                                             average='weighted')
        final_precision = final_precision + precision
        final_recall = final_recall + recall
        final_fscore = final_fscore + fscore


    return [final_accuracy/k,
            final_precision/k,
            final_recall/k,
            final_fscore/k]

def loss(y, y_pred):
    result = np.sum(y == y_pred)/float(len(y_pred))
    print("--------Current Accuracy: " + str(result))
    return result