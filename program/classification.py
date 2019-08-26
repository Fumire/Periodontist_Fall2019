import pandas
import numpy
import xgboost
import data


def get_weights(file_name, level=4):
    """
    get weights of feature importances
    last modified: 2019-08-26T16:24:57+0900
    """
    raw_data = data.processed_data(file_name, level=level)

    x_train = raw_data.drop(columns=["classification"])
    y_train = raw_data["classification"]

    print(x_train.columns)

    model = xgboost.XGBClassifier(n_jobs=-1, random_state=0)
    model.fit(x_train, y_train)

    return model.feature_importances_


if __name__ == "__main__":
    print(get_weights("1.tsv"))
