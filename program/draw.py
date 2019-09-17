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
    last modified: 2019-08-29T18:16:18+0900
    """
    raw_data = data.processed_data(file_name)
    tsne_data = data.get_tsne(file_name)

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 30})

    matplotlib.pyplot.figure()
    for color, item in zip(["g", "b", "r", "k"], ["H", "CPE", "CPM", "CPS"]):
        selected_data = tsne_data.loc[raw_data["classification"] == item]
        matplotlib.pyplot.scatter(selected_data["TSNE-1"], selected_data["TSNE-2"], c=color, marker="$%s$" % item[-1], s=200, label=item)

    matplotlib.pyplot.title("TSNE map: " + file_name)
    matplotlib.pyplot.xlabel("Standardized TSNE-1")
    matplotlib.pyplot.ylabel("Standardized TSNE-2")
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.legend()

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


def draw_prediction(function, file_name, level, k_fold=5):
    """
    draw prediction
    reference: https://scikit-learn.org/stable/auto_examples/svm/plot_custom_kernel.html#sphx-glr-auto-examples-svm-plot-custom-kernel-py
    last modified: 2019-08-29T18:39:05+0900
    """
    best, level = classification.run_test(function, file_name, level, k_fold=k_fold)
    y_answer, y_predict, y_index = function(file_name=file_name, number=best, level=level, return_score=False, k_fold=k_fold)
    score = function(file_name=file_name, number=best, level=level, k_fold=k_fold)
    tsne = data.get_tsne(file_name)
    label = {"H": "g", "CPE": "b", "CPM": "r", "CPS": "k"}

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 30})

    matplotlib.pyplot.figure()
    for i in y_index:
        matplotlib.pyplot.scatter(tsne.iloc[i, 0], tsne.iloc[i, 1], c=label[y_predict.iloc[i]], s=200, marker="$%s$" % y_answer.iloc[i][-1])

    matplotlib.pyplot.title(function.__name__ + ": %.2f" % score)
    matplotlib.pyplot.xlabel("Standardized TSNE-1")
    matplotlib.pyplot.ylabel("Standardized TSNE-2")
    matplotlib.pyplot.grid(True)

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(24, 24)
    fig.savefig("figures/" + "Prediction" + current_time() + ".png")


def draw_prediction_binary(function, file_name, level, k_fold=5):
    """
    draw prediction in OX
    last modified: 2019-08-29T18:38:56+0900
    """
    best, level = classification.run_test(function, file_name, level, k_fold=k_fold)
    y_answer, y_predict, y_index = function(file_name=file_name, number=best, level=level, return_score=False, k_fold=k_fold)
    score = function(file_name=file_name, number=best, level=level, k_fold=k_fold)
    tsne = data.get_tsne(file_name)
    label = {"H": "g", "CPE": "b", "CPM": "r", "CPS": "k"}

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 30})

    matplotlib.pyplot.figure()
    for i in y_index:
        if y_predict.iloc[i] == y_answer.iloc[i]:
            matplotlib.pyplot.scatter(tsne.iloc[i, 0], tsne.iloc[i, 1], c=label[y_answer.iloc[i]], s=200, marker="o")
        else:
            matplotlib.pyplot.scatter(tsne.iloc[i, 0], tsne.iloc[i, 1], c=label[y_answer.iloc[i]], s=200, marker="X")

    matplotlib.pyplot.title(function.__name__ + ": %.2f" % score)
    matplotlib.pyplot.xlabel("Standardized TSNE-1")
    matplotlib.pyplot.ylabel("Standardized TSNE-2")
    matplotlib.pyplot.grid(True)

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(24, 24)
    fig.savefig("figures/" + "PredictionOX" + current_time() + ".png")


def draw_binary_prediction(function, file_name, level, group_list, k_fold=5):
    """
    binary prediction which is or which is not
    last modified: 2019-09-06T16:09:40+0900
    """
    best, level = classification.run_test(function, file_name, level, group_list=group_list, k_fold=k_fold)
    y_answer, y_predict, y_index = function(file_name=file_name, number=best, level=level, return_score=False, k_fold=k_fold, group_list=group_list)
    score = function(file_name=file_name, number=best, level=level, return_score=True, k_fold=k_fold, group_list=group_list)
    tsne = data.get_tsne(file_name)
    color = ["r", "b"]

    assert len(set(y_predict.tolist())) == 2

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 30})

    matplotlib.pyplot.figure()
    for i, prediction in enumerate(list(set(y_predict.tolist()))):
        index = list(filter(lambda x: list((y_predict == prediction) & (y_predict == y_answer))[x], y_index))
        matplotlib.pyplot.scatter(tsne.iloc[index, 0], tsne.iloc[index, 1], c=color[i], s=200, marker="o", label=prediction)

        index = list(filter(lambda x: list((y_predict == prediction) & (y_predict != y_answer))[x], y_index))
        matplotlib.pyplot.scatter(tsne.iloc[index, 0], tsne.iloc[index, 1], c=color[i], s=200, marker="X", label=prediction)

    matplotlib.pyplot.title(function.__name__ + ": %.2f" % score)
    matplotlib.pyplot.xlabel("Standardized TSNE-1")
    matplotlib.pyplot.ylabel("Standardized TSNE-2")
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.legend()

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(24, 24)
    fig.savefig("figures/" + "BinaryPrediction" + current_time() + ".png")


def draw_with_best_combination(file_name, function, level=6, k_fold=5, group_list=["H", "CPE", "CPM", "CPS"]):
    """
    draw the best combination of features
    last modified: 
    """
    best_combination = classification.scoring_with_best_combination(file_name, function, level, k_fold, group_list)
    score = list(map(lambda x: x[1], best_combination))

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 30})

    matplotlib.pyplot.figure()
    matplotlib.pyplot.boxplot(score)

    matplotlib.pyplot.title("Best combination: " + file_name)

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(48, 24)
    fig.savefig("figures/" + "BestCombination" + current_time() + ".png")


if __name__ == "__main__":
    for file_name in ["1.tsv", "2.tsv"]:
        draw_tsne_with_marker(file_name)
        for function in [classification.classification_with_SVC, classification.classification_with_XGBClassifier, classification.classification_with_KNeighbors, classification.classification_with_RandomForest]:
            draw_prediction(function, file_name, 5)
            draw_prediction_binary(function, file_name, 5)
            for grouping in [["H", "Not_H", "Not_H", "Not_H"], ["Not_S", "Not_S", "Not_S", "S"], ["H&E", "H&E", "M&S", "M&S"]]:
                draw_binary_prediction(function, file_name, 5, grouping)
                draw_with_best_combination(file_name, function, group_list=grouping)
