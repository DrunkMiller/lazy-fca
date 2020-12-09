import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
import online_classification

def print_all_accuracy_metris(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    print("Accuracy: {}\nROC AUC: {}".format(acc, roc_auc))
    TP = np.sum(y_test * y_pred)
    TN = np.sum(y_test + y_pred == 0)
    FP = np.sum((y_test == 0) * (y_pred == 1))
    FN = np.sum((y_test == 1) * (y_pred == 0))
    TPR = float(TP) / (TP + FN)
    TNR = float(TN) / (TN + FP)
    FPR = float(FP) / (FP + TN)
    NPV = float(TN) / (TN + FN)
    FDR = float(FP) / (TP + FP)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    print('True Positive: {}'.format(TP))
    print('True Negative: {} '.format(TN))
    print('False Positive: {}'.format(FP))
    print('False Negative: {}'.format(FN))
    print('True Positive Rate: {}'.format(TPR))
    print('True Negative Rate: {}'.format(TNR))
    print('Negative Predictive Value: {}'.format(NPV))
    print('False Positive Rate: {}'.format(FPR))
    print('False Discovery Rate: {}'.format(FDR))
    print('Precision: {}'.format(prec))
    print('Recall: {}'.format(rec))


def prepare_data(test_df, train_df):
    x_test_df = test_df.copy()
    x_test_df.drop('credit_approval', axis='columns', inplace=True)
    y_test_df = test_df['credit_approval']

    plus_df = train_df[train_df['credit_approval'] == 1]
    minus_df = train_df[train_df['credit_approval'] == 0]
    plus_df.drop('credit_approval', axis='columns', inplace=True)
    minus_df.drop('credit_approval', axis='columns', inplace=True)

    return plus_df.values.tolist(), minus_df.values.tolist(), x_test_df.values.tolist(), y_test_df.values.tolist()


def cross_objects(list_one, list_two):
    cross = []
    for i in range(len(list_one)):
        cross += [list_one[i] * list_two[i]]
    return cross


def algorithm1(plus, minus, x_pred):
    y_pred = []
    for i in x_pred:
        positive = 0
        negative = 0
        for j in plus:
            res = cross_objects(i, j)
            positive += sum(res) / len(j)
        positive = float(positive) / len(plus)
        for j in minus:
            res = cross_objects(i, j)
            negative += sum(res) / len(j)
        negative = float(negative) / len(minus)
        if (positive > negative):
            y_pred.append(1)
        else:
            y_pred.append(0)
    return y_pred


def calc_count_cross(crossed, inter):
    count = 0
    for cross in crossed:
        if sum(cross_objects(cross, inter)) == sum(inter):
            count += 1
    return count


def algorithm2(plus, minus, x_test, threshold_coef=1):
    y_pred = []
    for i in x_test:
        positive_voices = 0
        negative_voices = 0
        for j in plus:
            cross = cross_objects(i, j)
            count_cross_with_minus = calc_count_cross(minus, cross)
            if count_cross_with_minus / len(minus) < threshold_coef:
                positive_voices += float(sum(cross)) / sum(j)

        for j in minus:
            cross = cross_objects(i, j)
            count_cross_with_plus = calc_count_cross(plus, cross)
            if count_cross_with_plus / len(plus) < threshold_coef:
                negative_voices += float(sum(cross)) / sum(j)

        if (positive_voices / len(plus) > negative_voices / len(minus)):
            y_pred.append(1)
        else:
            y_pred.append(0)
    return y_pred


def other_classiffiers(data):
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=33, shuffle=True)

    x_test = test_df.copy()
    x_test.drop('credit_approval', axis='columns', inplace=True)

    x_train = train_df.copy()
    x_train.drop('credit_approval', axis='columns', inplace=True)

    y_train = train_df['credit_approval']
    y_test = test_df['credit_approval']

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(x_train)
    X_train = scaler.transform(x_train)
    X_test = scaler.transform(x_test)

    print('---------------------------------')
    print('#            Naive Bayes        #')
    print('---------------------------------')
    from sklearn.naive_bayes import GaussianNB
    GNBclassifier = GaussianNB()
    GNBclassifier.fit(X_train, y_train)
    y_pred1 = GNBclassifier.predict(X_test)
    print_all_accuracy_metris(np.array(y_test), np.array(y_pred1))

    print('---------------------------------')
    print('#            K Neighbors        #')
    print('---------------------------------')
    from sklearn.neighbors import KNeighborsClassifier
    KNclassifier = KNeighborsClassifier(n_neighbors=5)
    KNclassifier.fit(X_train, y_train)
    y_pred2 = KNclassifier.predict(X_test)
    print_all_accuracy_metris(np.array(y_test), np.array(y_pred2))

    print('-----------------------------------')
    print('#            Decision Tree        #')
    print('-----------------------------------')
    from sklearn.tree import DecisionTreeClassifier
    DTclassifier = DecisionTreeClassifier()
    DTclassifier.fit(X_train, y_train)
    y_pred3 = DTclassifier.predict(X_test)
    print_all_accuracy_metris(np.array(y_test), np.array(y_pred3))


if __name__== "__main__":
    data = pd.read_csv('binarized_credit.csv', delimiter=';')
    features_names = (list(data))
    features_names.remove('credit_approval')

    train_df, test_df = train_test_split(data, test_size=0.2, random_state=33, shuffle=True)
    plus, minus, x_test, y_test = prepare_data(test_df, train_df)

    other_classiffiers(data)

    print('---------------------------------')
    print('#            Algorithm 1        #')
    print('---------------------------------')
    algorithm1_y_pred = algorithm1(plus, minus, x_test)
    print_all_accuracy_metris(np.array(y_test), np.array(algorithm1_y_pred))
    print('---------------------------------')
    print('#            Algorithm 2        #')
    print('---------------------------------')
    algorithm2_y_pred = algorithm2(plus, minus, x_test, 0.5)
    print_all_accuracy_metris(np.array(y_test), np.array(algorithm2_y_pred))
    print('---------------------------------')
    print('#      Online Algorithm 1       #')
    print('---------------------------------')
    online = online_classification.OnlineClassifier(plus, minus, algorithm1)
    online_y_pred1 = online.predict(x_test)
    print_all_accuracy_metris(np.array(y_test), np.array(online_y_pred1))
    print('---------------------------------')
    print('#      Online Algorithm 2       #')
    print('---------------------------------')
    online = online_classification.OnlineClassifier(plus, minus, lambda p, m, t: algorithm2(p, m, t, 0.5))
    online_y_pred2 = online.predict(x_test)
    print_all_accuracy_metris(np.array(y_test), np.array(online_y_pred2))