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


def get_features(file_name, level=6, limitation=True):
    """
    get features in order of importances. Remove feature importances is zero.
    last modified: 2019-09-18T00:48:13+0900
    """
    if limitation:
        raw_features = get_feature_importances(file_name, level=level)
    else:
        raw_features = get_feature_importances(file_name, level=level, percentile=None, top=None)
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


def change_feature_to_number(file_name, features, level=6):
    """
    change feature to number
    last modified: 2019-09-17T23:54:03+0900
    """
    raw_features = get_features(file_name, level=level)
    number = 0

    for i, feature in enumerate(features):
        number += 2 ** raw_features.index(feature)

    return number


def run_test(function, file_name, level, processes=100, k_fold=5, group_list=["H", "CPE", "CPM", "CPS"]):
    """
    execute test for given function.
    last modified: 2019-09-06T14:47:29+0900
    """
    features = get_features(file_name, level)
    print(len(features))

    with multiprocessing.Pool(processes=processes) as pool:
        score = pool.starmap(function, [(file_name, i, level, True, k_fold, group_list) for i in range(1, 2 ** len(features))])
    best = int(numpy.argmax(score)) + 1
    combination = sorted(change_number_to_feature(file_name, best, level))

    print(function.__name__, max(score), best, level, combination)

    return best, level


def classification_with_XGBClassifier(file_name, number, level, return_score=True, k_fold=5, group_list=["H", "CPE", "CPM", "CPS"]):
    """
    classification with XGBClassifier
    last modified: 2019-09-06T14:32:52+0900
    """
    _pickle_file = "pickles/classification_with_XGBClassifier_" + file_name + "_" + str(number) + "_" + str(level) + "_" + str(k_fold) + "_" + "+".join(group_list) + ".pkl"
    if os.path.exists(_pickle_file):
        with open(_pickle_file, "rb") as f:
            y_answer, y_predict, y_index = pickle.load(f)
    else:
        raw_data = data.processed_data(file_name=file_name, level=level, group_list=group_list)

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


def classification_with_SVC(file_name, number, level, return_score=True, k_fold=5, group_list=["H", "CPE", "CPM", "CPS"]):
    """
    classification with SVC of SVM
    reference: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    last modified: 2019-09-06T14:34:22+0900
    """
    _pickle_file = "pickles/classification_with_SVC_" + file_name + "_" + str(number) + "_" + str(level) + "_" + str(k_fold) + "_" + "+".join(group_list) + ".pkl"
    if os.path.exists(_pickle_file):
        with open(_pickle_file, "rb") as f:
            y_answer, y_predict, y_index = pickle.load(f)

        if return_score:
            return numpy.mean(list(map(lambda x: 1 if x[0] == x[1] else 0, zip(y_answer, y_predict, y_index))))
        else:
            return y_answer, y_predict, y_index
    else:
        raw_data = data.processed_data(file_name=file_name, level=level, group_list=group_list)

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


def classification_with_KNeighbors(file_name, number, level, return_score=True, k_fold=5, group_list=["H", "CPE", "CPM", "CPS"]):
    """
    classification with Nearest Neighbors algorithm
    reference: https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#sphx-glr-auto-examples-neighbors-plot-classification-py
    last modified: 2019-09-06T14:34:42+0900
    """
    _pickle_file = "pickles/KNeighbors_" + file_name + "_" + str(number) + "_" + str(level) + "_" + str(k_fold) + "_" + "+".join(group_list) + ".pkl"
    if os.path.exists(_pickle_file):
        with open(_pickle_file, "rb") as f:
            y_answer, y_predict, y_index = pickle.load(f)

        if return_score:
            return numpy.mean(list(map(lambda x: 1 if x[0] == x[1] else 0, zip(y_answer, y_predict, y_index))))
        else:
            return y_answer, y_predict, y_index
    else:
        raw_data = data.processed_data(file_name=file_name, level=level, group_list=group_list)

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


def classification_with_RandomForest(file_name, number, level, return_score=True, k_fold=5, group_list=["H", "CPE", "CPM", "CPS"]):
    """
    classification with RandomForest
    reference: https://scikit-learn.org/stable/modules/ensemble.html#forest
    last modified: 2019-09-06T14:35:18+0900
    """
    _pickle_file = "pickles/classification_with_RandomForest_" + file_name + "_" + str(number) + "_" + str(level) + str(k_fold) + "_" + "+".join(group_list) + ".pkl"
    if os.path.exists(_pickle_file):
        with open(_pickle_file, "rb") as f:
            y_answer, y_predict, y_index = pickle.load(f)

        if return_score:
            return numpy.mean(list(map(lambda x: 1 if x[0] == x[1] else 0, zip(y_answer, y_predict, y_index))))
        else:
            return y_answer, y_predict, y_index
    else:
        raw_data = data.processed_data(file_name=file_name, level=level, group_list=group_list)

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


def scoring_with_best_combination(file_name, function, level=6, k_fold=5, group_list=["H", "CPE", "CPM", "CPS"]):
    """
    Find best combination of features
    last modified: 2019-09-18T00:17:30+0900
    """
    _pickle_file = "pickles/best_combination_" + file_name + "_" + function.__name__ + "_" + str(level) + "_" + str(k_fold) + "_" + "+".join(group_list) + ".pkl"
    if os.path.exists(_pickle_file):
        with open(_pickle_file, "rb") as f:
            best_combination = pickle.load(f)
    else:
        best_combination = [(list(), 0)]
        raw_features = get_features(file_name, level=level, limitation=False)
        with multiprocessing.Pool(processes=100) as pool:
            for _ in raw_features:
                using_combination = list(map(lambda x: change_feature_to_number(file_name, x, level), [best_combination[-1][0] + [x] for x in list(filter(lambda x: x not in best_combination[-1][0], raw_features))]))
                score = pool.starmap(function, [(file_name, number, level, True, k_fold, group_list) for number in using_combination])
                best_combination.append(((change_number_to_feature(file_name, using_combination[int(numpy.argmax(score))])), sorted(score)))

        best_combination = best_combination[1:]
        with open(_pickle_file, "wb") as f:
            pickle.dump(best_combination, f)

    return best_combination


if __name__ == "__main__":
    for file_name in ["1.tsv", "2.tsv"]:
        for function in [classification_with_XGBClassifier, classification_with_SVC, classification_with_KNeighbors, classification_with_RandomForest]:
            scoring_with_best_combination(file_name, function)
            best, level = run_test(function, file_name, 5, group_list=["H", "Not_H", "Not_H", "Not_H"])
