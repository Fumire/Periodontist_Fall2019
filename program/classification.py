import multiprocessing
import os
import pickle
import pandas
import numpy
import sklearn
import sklearn.ensemble
import xgboost
import data


def get_feature_importances(file_name, level=6, percentile=None, top=10):
    """
    get feature importances
    last modified: 2019-08-29T14:06:01+0900
    """
    _pickle_file = "pickles/feature_importances_" + file_name + "_" + str(level) + ".pkl"
    if os.path.exists(_pickle_file):
        with open(_pickle_file, "rb") as f:
            return_data = pickle.load(f)
    else:
        raw_data = data.processed_data(file_name, level=level)

        x_train = raw_data.drop(columns=["classification"])
        y_train = raw_data["classification"]

        model = xgboost.XGBClassifier(n_jobs=-1, random_state=0)
        model.fit(x_train, y_train)

        return_data = sorted(list(zip(model.feature_importances_, list(raw_data.columns))), reverse=True)

        with open(_pickle_file, "wb") as f:
            pickle.dump(return_data, f)

    if percentile is None and top is None:
        return return_data
    elif percentile is not None and top is not None:
        raise ValueError
    elif percentile is None and top is not None:
        return return_data[:top]
    elif percentile is not None and top is None:
        return return_data[:len(return_data) * percentile // 100]
    else:
        raise ValueError


def get_features(file_name, level=6):
    """
    get features in order of importances. Remove feature importances is zero.
    last modified: 2019-08-27T17:13:38+0900
    """
    raw_features = get_feature_importances(file_name, level=level)
    raw_features = list(filter(lambda x: x[0], raw_features))

    return list(map(lambda x: x[1], raw_features))


def change_number_to_feature(file_name, number, level=6):
    """
    change number to feature
    last modified: 2019-08-27T17:12:07+0900
    """
    raw_features = get_features(file_name, level=level)

    if number > 2 ** len(raw_features):
        raise ValueError

    check_list = list(map(lambda x: ((2 ** x) & number) != 0, range(2 ** len(raw_features))))

    return list(filter(lambda x: check_list[raw_features.index(x)], raw_features))


def run_test(function, file_name, level, processes=100, k_fold=5):
    """
    execute test for given function.
    last modified: 2019-08-29T16:29:14+0900
    """
    features = get_features(file_name, level)
    print(len(features))

    with multiprocessing.Pool(processes=processes) as pool:
        score = pool.starmap(function, [(file_name, i, level, True, k_fold) for i in range(1, 2 ** len(features))])
    best = int(numpy.argmax(score)) + 1
    combination = sorted(change_number_to_feature(file_name, best, level))

    print(function.__name__, max(score), best, level, combination)

    return best, level


def classification_with_XGBClassifier(file_name, number, level, return_score=True, k_fold=5):
    """
    classification with XGBClassifier
    last modified: 2019-08-29T16:54:18+0900
    """
    _pickle_file = "pickles/classification_with_XGBClassifier_" + file_name + "_" + str(number) + "_" + str(level) + "_" + str(k_fold) + ".pkl"
    if os.path.exists(_pickle_file):
        with open(_pickle_file, "rb") as f:
            y_answer, y_predict, y_index = pickle.load(f)
    else:
        raw_data = data.processed_data(file_name=file_name, level=level)

        selected_features = change_number_to_feature(file_name, number, level)
        x_data = raw_data[selected_features]
        y_data = raw_data[["classification"]]

        y_answer, y_predict, y_index = list(), list(), list()

        kf = sklearn.model_selection.KFold(n_splits=k_fold, random_state=0)
        for train_data_index, test_data_index in kf.split(X=x_data, y=y_data):
            x_train, y_train = x_data.iloc[train_data_index, :], numpy.ravel(y_data.iloc[train_data_index, :].values.tolist(), order="C")
            x_test, y_test = x_data.iloc[test_data_index, :], numpy.ravel(y_data.iloc[test_data_index, :].values.tolist(), order="C")

            model = xgboost.XGBClassifier(n_jobs=1, random_state=0)
            model.fit(x_train, y_train, verbose=False, eval_set=[(x_test, y_test)])

            y_answer.append(pandas.Series(y_test))
            y_predict.append(pandas.Series(model.predict(x_test)))
            y_index += list(test_data_index)

        y_answer, y_predict = pandas.concat(y_answer, ignore_index=False), pandas.concat(y_predict, ignore_index=False)

        with open(_pickle_file, "wb") as f:
            pickle.dump((y_answer, y_predict, y_index), f)

    if return_score:
        return numpy.mean(list(map(lambda x: 1 if x[0] == x[1] else 0, zip(y_answer, y_predict, y_index))))
    else:
        return y_answer, y_predict, y_index


def classification_with_SVC(file_name, number, level, return_score=True, k_fold=5, verbose=False):
    """
    classification with SVC of SVM
    reference: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    last modified: 2019-09-03T17:39:30+0900
    """
    _pickle_file = "pickles/classification_with_SVC_" + file_name + "_" + str(number) + "_" + str(level) + "_" + str(k_fold) + ".pkl"
    if os.path.exists(_pickle_file):
        with open(_pickle_file, "rb") as f:
            y_answer, y_predict, y_index = pickle.load(f)

        if return_score:
            return numpy.mean(list(map(lambda x: 1 if x[0] == x[1] else 0, zip(y_answer, y_predict, y_index))))
        else:
            return y_answer, y_predict, y_index
    else:
        raw_data = data.processed_data(file_name=file_name, level=level)

        selected_features = change_number_to_feature(file_name, number, level)
        x_data = raw_data[selected_features]
        y_data = raw_data[["classification"]]

        y_answer, y_predict, y_index = list(), list(), list()

        kf = sklearn.model_selection.KFold(n_splits=k_fold, random_state=0)
        for train_data_index, test_data_index in kf.split(X=x_data, y=y_data):
            x_train, y_train = x_data.iloc[train_data_index, :], numpy.ravel(y_data.iloc[train_data_index, :].values.tolist(), order="C")
            x_test, y_test = x_data.iloc[test_data_index, :], numpy.ravel(y_data.iloc[test_data_index, :].values.tolist(), order="C")

            clf = sklearn.svm.SVC(random_state=0, gamma="scale")
            clf.fit(x_train, y_train)

            y_answer.append(pandas.Series(y_test))
            y_predict.append(pandas.Series(clf.predict(x_test)))
            y_index += list(test_data_index)

        y_answer, y_predict = pandas.concat(y_answer, ignore_index=False), pandas.concat(y_predict, ignore_index=False)

        with open(_pickle_file, "wb") as f:
            pickle.dump((y_answer, y_predict, y_index), f)

        if return_score:
            return numpy.mean(list(map(lambda x: 1 if x[0] == x[1] else 0, zip(y_answer, y_predict))))
        else:
            return y_answer, y_predict, y_index


def classification_with_KNeighbors(file_name, number, level, return_score=True, k_fold=5, verbose=False):
    """
    classification with Nearest Neighbors algorithm
    reference: https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#sphx-glr-auto-examples-neighbors-plot-classification-py
    last modified: 2019-09-03T17:38:27+0900
    """
    _pickle_file = "pickles/KNeighbors_" + file_name + "_" + str(number) + "_" + str(level) + "_" + str(k_fold) + ".pkl"
    if os.path.exists(_pickle_file):
        with open(_pickle_file, "rb") as f:
            y_answer, y_predict, y_index = pickle.load(f)

        if return_score:
            return numpy.mean(list(map(lambda x: 1 if x[0] == x[1] else 0, zip(y_answer, y_predict, y_index))))
        else:
            return y_answer, y_predict, y_index
    else:
        raw_data = data.processed_data(file_name=file_name, level=level)

        selected_features = change_number_to_feature(file_name, number, level)
        x_data, y_data = raw_data[selected_features], raw_data[["classification"]]

        y_answer, y_predict, y_index = list(), list(), list()
        kf = sklearn.model_selection.KFold(n_splits=k_fold, random_state=0)
        for train_data_index, test_data_index in kf.split(X=x_data, y=y_data):
            x_train, y_train = x_data.iloc[train_data_index, :], numpy.ravel(y_data.iloc[train_data_index, :].values.tolist(), order="C")
            x_test, y_test = x_data.iloc[test_data_index, :], numpy.ravel(y_data.iloc[test_data_index, :].values.tolist(), order="C")

            clf = sklearn.neighbors.KNeighborsClassifier()
            clf.fit(x_train, y_train)

            y_answer.append(pandas.Series(y_test))
            y_predict.append(pandas.Series(clf.predict(x_test)))
            y_index += list(test_data_index)

        y_answer, y_predict = pandas.concat(y_answer, ignore_index=False), pandas.concat(y_predict, ignore_index=False)

        with open(_pickle_file, "wb") as f:
            pickle.dump((y_answer, y_predict, y_index), f)

        if return_score:
            return numpy.mean(list(map(lambda x: 1 if x[0] == x[1] else 0, zip(y_answer, y_predict, y_index))))
        else:
            return y_answer, y_predict, y_index


def classification_with_RandomForest(file_name, number, level, return_score=True, k_fold=5, verbose=False):
    """
    classification with RandomForest
    reference: https://scikit-learn.org/stable/modules/ensemble.html#forest
    last modified: 2019-09-03T17:50:22+0900
    """
    _pickle_file = "pickles/classification_with_RandomForest_" + file_name + "_" + str(number) + "_" + str(level) + str(k_fold) + ".pkl"
    if os.path.exists(_pickle_file):
        with open(_pickle_file, "rb") as f:
            y_answer, y_predict, y_index = pickle.load(f)

        if return_score:
            return numpy.mean(list(map(lambda x: 1 if x[0] == x[1] else 0, zip(y_answer, y_predict, y_index))))
        else:
            return y_answer, y_predict, y_index
    else:
        raw_data = data.processed_data(file_name=file_name, level=level)

        x_data, y_data = raw_data[change_number_to_feature(file_name, number, level)], raw_data[["classification"]]

        y_answer, y_predict, y_index = list(), list(), list()

        kf = sklearn.model_selection.KFold(n_splits=k_fold, random_state=0)
        for train_data_index, test_data_index in kf.split(X=x_data, y=y_data):
            x_train, y_train = x_data.iloc[train_data_index, :], numpy.ravel(y_data.iloc[train_data_index, :].values.tolist(), order="C")
            x_test, y_test = x_data.iloc[test_data_index, :], numpy.ravel(y_data.iloc[test_data_index, :].values.tolist(), order="C")

            clf = sklearn.ensemble.RandomForestClassifier(random_state=0, n_estimators=100)
            clf.fit(x_train, y_train)

            y_answer.append(pandas.Series(y_test))
            y_predict.append(pandas.Series(clf.predict(x_test)))
            y_index += list(test_data_index)

        y_answer, y_predict = pandas.concat(y_answer, ignore_index=False), pandas.concat(y_predict, ignore_index=False)

        with open(_pickle_file, "wb") as f:
            pickle.dump((y_answer, y_predict, y_index), f)

        if return_score:
            return numpy.mean(list(map(lambda x: 1 if x[0] == x[1] else 0, zip(y_answer, y_predict))))
        else:
            return y_answer, y_predict, y_index


if __name__ == "__main__":
    for file_name in ["1.tsv", "2.tsv"]:
        for function in [classification_with_XGBClassifier, classification_with_SVC, classification_with_KNeighbors, classification_with_RandomForest]:
            best, level = run_test(function, file_name, 5)
