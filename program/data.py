import os
import pickle
import pandas
import scipy
import sklearn
import sklearn.manifold

"""
get and save data with this module.:w

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
    calculate tsne from file, and save data into pickle."
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


def processed_data(file_name):
    """
    return proceesed data which is ready to use
    last modified: 2019-08-26T11:54:06+0900
    """
    data = get_data(file_name)
    data = make_class_column(data)

    return data


if __name__ == "__main__":
    for file_name in ["1.tsv", "2.tsv"]:
        print(get_tsne(file_name))
