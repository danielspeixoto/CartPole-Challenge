import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import metrics
import pandas as pd

def k_fold(k, df, y, clf):
    fold = StratifiedKFold(k)
    final_accuracy = 0.0
    final_precision = 0.0
    final_recall = 0.0
    final_fscore = 0.0
    for train, test in fold.split(df, y):
        X_train, X_test = list(pd.Series(df)[train]), list(pd.Series(df)[test])
        y_train, y_test = list(pd.Series(y)[train]), list(pd.Series(y)[test])
        clf = clf.fit(X_train, y_train)
        predicted = np.round(np.clip(clf.predict(X_test), 0, 1))
        accuracy = np.mean(predicted == y_test)
        final_accuracy = final_accuracy + accuracy
        precision = metrics.precision_score(y_test, predicted)
        final_precision = final_precision + precision
        recall = metrics.recall_score(y_test, predicted)
        final_recall = final_recall + recall
        fscore = metrics.f1_score(y_test, predicted)
        final_fscore = final_fscore + fscore


    return [final_accuracy/k,
            final_precision/k,
            final_recall/k,
            final_fscore/k,]

def loss(y, y_pred):
    result = np.sum(y == y_pred)/float(len(y_pred))
    print("--------Current Accuracy: " + str(result))
    return result