import time
import matplotlib
import matplotlib.pyplot
import numpy
import classification
import data

"""
draw any data into PNG
"""


def current_time():
    """
    get current time for file name.
    Last modified: 2019-08-26T12:56:35+0900
    """
    time.sleep(1)
    return "_" + time.strftime("%m%d%H%M%S")


def draw_tsne(file_name):
    """
    draw TSNE plot.
    Last modified: 2019-08-22T21:02:55+0900
    """
    raw_data = data.get_tsne(file_name)

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 30})

    matplotlib.pyplot.figure()
    matplotlib.pyplot.scatter(raw_data["TSNE-1"], raw_data["TSNE-2"])

    matplotlib.pyplot.title("TSNE map: " + file_name)
    matplotlib.pyplot.xlabel("Standardized TSNE-1")
    matplotlib.pyplot.ylabel("Standardized TSNE-2")
    matplotlib.pyplot.grid(True)

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(24, 24)
    fig.savefig("figures/" + "TSNE" + current_time() + ".png")


def draw_tsne_with_marker(file_name):
    """
    draw TSNE plot with marker.
    last modified: 2019-08-26T14:05:52+0900
    """
    raw_data = data.processed_data(file_name)
    tsne_data = data.get_tsne(file_name)

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 30})

    matplotlib.pyplot.figure()
    for color, item in zip(["0.25", "0.50", "0.75", "1.00"], ["H", "CPE", "CPM", "CPS"]):
        selected_data = tsne_data.loc[raw_data["classification"] == item]
        matplotlib.pyplot.scatter(selected_data["TSNE-1"], selected_data["TSNE-2"], c=color, edgecolor='k', label=item)

    matplotlib.pyplot.title("TSNE map: " + file_name)
    matplotlib.pyplot.xlabel("Standardized TSNE-1")
    matplotlib.pyplot.ylabel("Standardized TSNE-2")
    matplotlib.pyplot.legend()
    matplotlib.pyplot.grid(True)

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(24, 24)
    fig.savefig("figures/" + "MarkerTSNE" + current_time() + ".png")


def draw_feature_importances(file_name, level=6):
    """
    draw feature importances plot.
    last modified: 2019-08-27T18:36:23+0900
    """
    raw_data = classification.get_feature_importances(file_name, level=level)

    values = list(map(lambda x: x[0], raw_data))

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 30})

    matplotlib.pyplot.figure()
    for i, value in enumerate(values):
        matplotlib.pyplot.bar(i, value)

    mean = numpy.mean(values)
    matplotlib.pyplot.plot([-1, len(values) + 1], [mean, mean], "k-", label="mean")

    for label, value, linestyle in zip(["75%", "50%", "25%"], numpy.percentile(values, [25, 50, 75]), ["--", "-.", ":", ""]):
        matplotlib.pyplot.plot([-1, len(values) + 1], [value, value], "k" + linestyle, label=label)

    matplotlib.pyplot.title("Feature Importances: " + file_name)
    matplotlib.pyplot.xlabel("Features")
    matplotlib.pyplot.ylabel("Importances")
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.legend()
    matplotlib.pyplot.xticks([])
    matplotlib.pyplot.ylim(0, 0.25)

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(32, 18)
    fig.savefig("figures/" + "FeatureImportances" + current_time() + ".png")


if __name__ == "__main__":
    for file_name in ["1.tsv", "2.tsv"]:
        for i in range(1, 7):
            draw_feature_importances(file_name, i)
            print(file_name, i)
