import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC


def importAndClean():
    df = pd.read_csv("HCMST_train.csv", header=0, low_memory=False)
    df_t = pd.read_csv("HCMST_test.csv", header=0, low_memory=False)

    df_train = df.drop('Q34', axis=1)
    df_test = df_t.drop('Q34', axis=1)

    df_to_train = df_train.apply(pd.to_numeric, errors='coerce')
    df_to_test = df_test.apply(pd.to_numeric, errors='coerce')

    # Drop any columns with NaNs
    df_to_train = df_to_train.dropna(axis=1, how='any')
    df_to_test = df_to_test.dropna(axis=1, how='any')


    x_train = df_to_train.values
    y_train = df['Q34'].values

    x_test = df_to_test.values
    y_test = df_t['Q34'].values

    return x_train, y_train, x_test, y_test

def applyClassifier(x,y,x_test):
    # Fit and Predict on New Data
    # In this case, predicting on training data b/c test_data has different dimensions for now...
    x_train_pred = OneVsRestClassifier(LinearSVC(random_state=0)).fit(x, y).predict(x)
    
    return x_train_pred

def main():
    x_train, y_train, x_test, y_test = importAndClean()
    pred = applyClassifier(x_train, y_train, x_test)
    
    print("Classifier Accuracy: %.2f" % np.mean(y_train == pred))

if __name__ == "__main__":
    main()
