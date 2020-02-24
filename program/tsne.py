import argparse
import os
import matplotlib
import matplotlib.pyplot
import pandas
import seaborn
import scipy
import sklearn.manifold
import general

default_tsne_directory = os.path.join(general.default_result_directory, "tsne")
tsne_columns = general.whole_values + ["AL", "PD", "DNA", "Total Bacteria"]


def get_tsne(csv_file=None, tsne_file=None, random_state=0):
    if tsne_file is None:
        tsne_file = os.path.join(default_tsne_directory, "tsne_" + str(random_state) + ".csv")

    if csv_file is None:
        raise ValueError
    elif not os.path.isfile(csv_file):
        raise ValueError

    data = pandas.read_csv(csv_file)

    tsne_data = pandas.DataFrame(sklearn.manifold.TSNE(n_components=2, random_state=random_state, init="pca").fit_transform(data[list(filter(lambda x: x in tsne_columns, list(data.columns)))]), columns=["TSNE1", "TSNE2"])
    for column in list(tsne_data.columns):
        tsne_data[column] = scipy.stats.zscore(tsne_data[column])
    for column in ["ID", "Classification"]:
        tsne_data[column] = data[column]

    tsne_data.to_csv(general.check_exist(tsne_file), index=False)
    return tsne_file


def draw_tsne(tsne_file=None, png_file=None):
    if png_file is None:
        png_file = os.path.join(default_tsne_directory, "tsne.png")

    if tsne_file is None:
        raise ValueError(tsne_file)
    elif not os.path.isfile(tsne_file):
        raise ValueError(tsne_file)

    seaborn.set(context="poster", style="whitegrid")

    fig, ax = matplotlib.pyplot.subplots(figsize=(24, 24))
    seaborn.scatterplot(data=pandas.read_csv(tsne_file), x="TSNE1", y="TSNE2", hue="Classification", style="Classification", legend="full", ax=ax)

    fig.savefig(general.check_exist(png_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-v", "--verbose", help="Verbose output", action="store_true", default=False)

    group1 = parser.add_mutually_exclusive_group(required=True)
    group1.add_argument("--calculate", help="Calculate TSNE", action="store_true", default=False)
    group1.add_argument("--draw", help="Draw TSNE", action="store_true", default=False)

    parser.add_argument("-i", "--input_file", help="File name of input", default=None)
    parser.add_argument("-o", "--output_file", help="File name to output", default=None)
    parser.add_argument("-r", "--random_state", help="Random state value", type=int, default=0)

    args = parser.parse_args()

    if args.calculate:
        data = get_tsne(csv_file=args.input_file, tsne_file=args.output_file, random_state=args.random_state)
        if args.verbose:
            print(data)
    elif args.draw:
        draw_tsne(tsne_file=args.input_file, png_file=args.output_file)
    else:
        exit("Something went wrong")
