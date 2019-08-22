import time
import matplotlib
import matplotlib.pyplot
import data


def current_time():
    """
    get current time for file name.
    Last modified: 2019-08-22T20:48:06+0900
    """
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


if __name__ == "__main__":
    for file_name in ["1.tsv", "2.tsv"]:
        draw_tsne(file_name)
        time.sleep(1)
