import multiprocessing
import os
import pickle
import pandas
import numpy
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
    last modified: 2019-08-29T13:26:22+0900
    """
    features = get_features(file_name, level)
    print(len(features))

    with multiprocessing.Pool(processes=processes) as pool:
        score = pool.starmap(function, [(file_name, i, level, True, k_fold) for i in range(1, 2 ** len(features))])
    best = int(numpy.argmax(score)) + 1
    combination = sorted(change_number_to_feature(file_name, best, level))

    print(max(score), best, level, combination)


def classification_with_XGBClassifier(file_name, number, level, return_score=True, k_fold=5, verbose=False):
    """
    classification with XGBClassifier
    last modified: 2019-08-29T13:40:37+0900
    """
    _pickle_file = "pickles/classification_with_XGBClassifier_" + file_name + "_" + str(number) + "_" + str(level) + "_" + str(k_fold) + ".pkl"
    if os.path.exists(_pickle_file):
        with open(_pickle_file, "rb") as f:
            y_answer, y_predict = pickle.load(f)
    else:
        folded_data = data.processed_data(file_name=file_name, level=level, for_validation=True, k_fold=k_fold)

        selected_features = change_number_to_feature(file_name, number, level)
        for i, elem in enumerate(folded_data):
            folded_data[i] = elem[selected_features + ["classification"]]

        y_answer, y_predict = list(), list()
        for i in range(k_fold):
            train_data = pandas.concat(list(map(lambda x: folded_data[i], list(filter(lambda x: x != i, range(k_fold))))))
            test_data = folded_data[i]

            if verbose:
                print(train_data)
                print(test_data)

            x_train, y_train = train_data.drop(columns=["classification"]), train_data["classification"]
            x_test, y_test = test_data.drop(columns=["classification"]), test_data["classification"]

            model = xgboost.XGBClassifier(n_jobs=1, random_state=0)
            model.fit(x_train, y_train, verbose=False, eval_set=[(x_test, y_test)])

            y_answer.append(y_test)
            y_predict.append(pandas.Series(model.predict(x_test)))

        y_answer, y_predict = pandas.concat(y_answer, ignore_index=True), pandas.concat(y_predict, ignore_index=True)

        with open(_pickle_file, "wb") as f:
            pickle.dump((y_answer, y_predict), f)

    if return_score:
        return numpy.mean(list(map(lambda x: 1 if x[0] == x[1] else 0, zip(y_answer, y_predict))))
    else:
        return y_predict


if __name__ == "__main__":
    run_test(classification_with_XGBClassifier, "1.tsv", 5)
    run_test(classification_with_XGBClassifier, "2.tsv", 5)
