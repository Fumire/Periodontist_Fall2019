import os
import pickle
import numpy
import pandas
import scipy
import sklearn
import sklearn.manifold

"""
get and save data with this module.
"""


def get_data(file_name):
    """
    get data from file.
    last modified: 2019-08-26T11:37:32+0900
    """
    data = pandas.read_csv("data/" + file_name, sep="\t", skiprows=1)

    pathogen_column = data["#OTU ID"].tolist()
    data.drop(columns=["#OTU ID"], inplace=True)

    data = data.transpose()
    data.columns = pathogen_column

    return data


def get_tsne(file_name):
    """
    calculate tsne from file, and save data into pickle.
    last modified: 2019-08-26T13:13:42+0900
    """
    _pickle_file = "pickles/tsne." + file_name + ".pkl"
    if os.path.exists(_pickle_file):
        with open(_pickle_file, "rb") as f:
            return pickle.load(f)
    else:
        raw_data = get_data(file_name)

        tsne = pandas.DataFrame(data=sklearn.manifold.TSNE(n_components=2, random_state=0).fit_transform(raw_data), columns=["TSNE-1", "TSNE-2"])
        tsne["TSNE-1"] = scipy.stats.zscore(tsne["TSNE-1"])
        tsne["TSNE-2"] = scipy.stats.zscore(tsne["TSNE-2"])
        tsne["id"] = raw_data.index

        tsne = tsne.set_index("id")

        with open(_pickle_file, "wb") as f:
            pickle.dump(tsne, f)

        return tsne


def make_class_column(raw_data):
    """
    make class column for classification
    last modified: 2019-08-26T13:19:44+0900
    """
    return raw_data.assign(classification=list(map(lambda x: x[:-3] if x[-3] == "1" else x[:-2], raw_data.index)))


def merge_columns(raw_data, level):
    """
    merge columns with level
    last modified: 2019-08-26T22:55:01+0900
    """
    if level < 0 or level > 6:
        raise ValueError

    if level == 6:
        return raw_data

    columns = list(map(lambda x: x.split(";"), raw_data.columns))
    for i, column in enumerate(columns):
        columns[i] = ";".join(column[:level + 1])
    columns = sorted(list(set(columns)))

    raw_columns = list(raw_data.columns)
    for column in columns:
        selected_columns = list(filter(lambda x: x.startswith(column), raw_columns))
        raw_data[column] = raw_data.loc[:, selected_columns].sum(axis=1)

    for column in raw_columns:
        del raw_data[column]

    return raw_data


def formatting_column(raw_data):
    """
    removing [, ], and < in column name
    last modifited: 2019-08-26T22:58:55+0900
    """
    columns = list(map(lambda x: x.replace("[", "").replace("]", "").replace("<", ""), list(raw_data.columns)))

    raw_data.columns = columns

    return raw_data


def drop_columns(raw_data):
    """
    drop columns which ends with "s__" or "__" such as "D_0__Bacteria;D_1__Bacteroidetes;D_2__Bacteroidia;D_3__Bacteroidales;D_4__Porphyromonadaceae;D_5__Porphyromonas;__"
    last modified: 2019-08-29T14:02:28+0900
    """
    for column in list(raw_data.columns):
        if column.split(";")[-1] in ["__", "s__"]:
            del raw_data[column]

    return raw_data


def processed_data(file_name, level=6, for_validation=False, k_fold=5):
    """
    return proceesed data which is ready to use
    last modified: 2019-08-29T14:02:53+0900
    """
    data = get_data(file_name)
    data = merge_columns(data, level)
    data = make_class_column(data)
    data = formatting_column(data)
    data = drop_columns(data)

    if for_validation:
        numpy.random.seed(0)
        mask = numpy.random.rand(len(data))

        return_data = list()
        for i in range(k_fold):
            return_data.append(data[numpy.all([(i / k_fold) <= mask, mask < ((i + 1) / k_fold)], axis=0)])
        return return_data
    else:
        return data


if __name__ == "__main__":
    for file_name in ["1.tsv", "2.tsv"]:
        for data in processed_data(file_name, 5):
            print(data.shape)
