import multiprocessing
import os
import pickle
import pandas
import numpy
import xgboost
import data


def get_feature_importances(file_name, level=6):
    """
    get feature importances
    last modified: 2019-08-26T23:34:51+0900
    """
    _pickle_file = "pickles/feature_importances_" + file_name + "_" + str(level) + ".pkl"
    if os.path.exists(_pickle_file):
        with open(_pickle_file, "rb") as f:
            return pickle.load(f)

    raw_data = data.processed_data(file_name, level=level)

    x_train = raw_data.drop(columns=["classification"])
    y_train = raw_data["classification"]

    model = xgboost.XGBClassifier(n_jobs=-1, random_state=0)
    model.fit(x_train, y_train)

    return_data = sorted(list(zip(model.feature_importances_, list(raw_data.columns))), reverse=True)

    with open(_pickle_file, "wb") as f:
        pickle.dump(return_data, f)

    return return_data


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


def run_test(function, file_name, level, processes=100):
    """
    execute test for given function.
    last modified: 2019-08-28T12:30:29+0900
    """
    features = get_features(file_name, level)
    print(len(features))

    with multiprocessing.Pool(processes=processes) as pool:
        score = pool.starmap(function, [(file_name, i, level, True) for i in range(1, 2 ** len(features))])
    best = int(numpy.argmax(score)) + 1
    combination = sorted(change_number_to_feature(file_name, best, level))

    print(max(score), best, combination)


def classification_with_XGBClassifier(file_name, number, level, return_score=True):
    """
    classification with XGBClassifier
    last modified: 2019-08-28T12:41:17+0900
    """
    _pickle_file = "pickles/classification_with_XGBClassifier_" + file_name + "_" + str(number) + "_" + str(level) + ".pkl"
    if os.path.exists(_pickle_file):
        with open(_pickle_file, "rb") as f:
            y_train, y_test = pickle.load(f)
    else:
        raw_data = data.processed_data(file_name=file_name, level=level)

        selected_features = change_number_to_feature(file_name, number, level)
        raw_data = raw_data[selected_features + ["classification"]]

        x_train = raw_data.drop(columns=["classification"])
        y_train = raw_data["classification"]

        model = xgboost.XGBClassifier(random_state=0)
        model.fit(x_train, y_train, verbose=False)

        y_test = model.predict(x_train)

        with open(_pickle_file, "wb") as f:
            pickle.dump((y_train, y_test), f)

    if return_score:
        return numpy.mean(list(map(lambda i: 1 if y_test[i] == y_train[i] else 0, range(len(y_test)))))
    else:
        return y_test


if __name__ == "__main__":
    run_test(classification_with_XGBClassifier, "1.tsv", 1)
    run_test(classification_with_XGBClassifier, "2.tsv", 1)
