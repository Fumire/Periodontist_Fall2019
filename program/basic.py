import os
import pickle
import pandas
import scipy
import sklearn
import sklearn.manifold


def get_data(file_name):
    """
    get data from file.
    last modified: 2019-08-21T11:30:57+0900
    """
    data = pandas.read_csv("data/" + file_name, sep="\t", skiprows=1)

    return data


def get_tsne(file_name):
    """
    calculate tsne from file, and save data into pickle."
    last modified
    """
    _pickle_file = "pickles/tsne." + file_name + ".pkl"
    if os.path.exists(_pickle_file):
        with open(_pickle_file, "rb") as f:
            return pickle.load(f)
    else:
        raw_data = get_data(file_name)

        ID_column = raw_data[["#OTU ID"]]
        raw_data.drop(["#OTU ID"], axis="columns", inplace=True)

        tsne = pandas.DataFrame(data=sklearn.manifold.TSNE(n_components=2, random_state=0).fit_transform(raw_data), columns=["TSNE-1", "TSNE-2"])
        tsne["TSNE-1"] = scipy.stats.zscore(tsne["TSNE-1"])
        tsne["TSNE-2"] = scipy.stats.zscore(tsne["TSNE-2"])
        tsne["id"] = ID_column

        with open(_pickle_file, "wb") as f:
            pickle.dump(tsne, f)

        return tsne


if __name__ == "__main__":
    for file_name in ["1.tsv", "2.tsv"]:
        print(get_data(file_name).columns)
        print(get_tsne(file_name))
