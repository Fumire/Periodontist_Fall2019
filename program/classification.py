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


if __name__ == "__main__":
    print(get_feature_importances("1.tsv"))
