import argparse
import itertools
import multiprocessing
import os
import sys
import matplotlib
import matplotlib.pyplot
import numpy
import pandas
import scipy
import sklearn.ensemble
import sklearn.gaussian_process
import sklearn.manifold
import sklearn.metrics
import sklearn.model_selection
import sklearn.neighbors
import sklearn.neural_network
import sklearn.tree

parser = argparse.ArgumentParser()

parser.add_argument("--verbose", help="Verbose output", action="store_true", default=False)
parser.add_argument("--file_name", help="File name to read data", type=str, default="/BiO/Store/Helixco/Periodontist_Fall2019/data/Periodontitis_input_dataset_from_784samples_and_additional_54samples_20190730.xlsx")
parser.add_argument("--include_ap", help="Include AP / does not", action="store_true", default=False)
parser.add_argument("--pickle_dir", help="Directory to store pickle data", type=str, default="results")
parser.add_argument("--png_dir", help="Directory to store PNG data", type=str, default="PNG")
parser.add_argument("--remake", help="Re-make from where", choices=range(101), default=100)
parser.add_argument("--random_state", help="Random number generator", type=int, default=0)
parser.add_argument("--KFold", help="Split number for KFold", type=int, default=5)
parser.add_argument("--jobs", help="Number of threads to use", type=int, default=50)

group1 = parser.add_mutually_exclusive_group(required=True)
group1.add_argument("--absolute", help="Use absolute values only", action="store_true", default=False)
group1.add_argument("--relative", help="Use relative values only", action="store_true", default=False)
group1.add_argument("--both", help="Use both absolute and relative values", action="store_true", default=False)

args = parser.parse_args()

remake_where = {"TSNE": 0, "four_groups": 1}
absolute_values = sorted(["Aa", "Pg", "Tf", "Td", "Pi", "Fn", "Pa", "Cr", "Ec"])
relative_values = sorted(["Aa_relative", "Pg_relative", "Tf_relative", "Td_relative", "Pi_relative", "Fn_relative", "Pa_relative", "Cr_relative", "Ec_relative"])
classes = ["H", "E", "M", "S"]
two_class_combinations = itertools.combinations(classes, 2)
three_class_combinations = itertools.combinations(classes, 3)
statistics = ("sensitivity", "specificity", "precision", "negative_predictive_value", "miss_rate", "fall_out", "false_discovery_rate", "false_ommission_rate", "thread_score", "accuracy", "F1_score", "odds_ratio")

multiclass_classifier_list = [("MLP", sklearn.neural_network.MLPClassifier(random_state=args.random_state, max_iter=2 ** 30, early_stopping=True)), ("SVC", sklearn.svm.SVC(decision_function_shape="ovr", random_state=args.random_state, probability=True)), ("KNeighbors", sklearn.neighbors.KNeighborsClassifier(n_jobs=1, algorithm="brute")), ("GaussianClassifier", sklearn.gaussian_process.GaussianProcessClassifier(max_iter_predict=2 ** 30, random_state=args.random_state, multi_class="one_vs_rest", n_jobs=1)), ("DecisionTree", sklearn.tree.DecisionTreeClassifier(random_state=args.random_state)), ("RandomForest", sklearn.ensemble.RandomForestClassifier(n_jobs=1, random_state=args.random_state)), ("AdaBoost", sklearn.ensemble.AdaBoostClassifier(random_state=args.random_state))]

using_features = list()
if args.absolute or args.both:
    using_features += absolute_values
if args.relative or args.both:
    using_features += relative_values
if using_features:
    using_features.sort()
else:
    exit("Something went wrong")

if not os.path.exists(args.file_name) or not os.path.isfile(args.file_name):
    exit("Invalid file: " + args.file)

data = pandas.concat(pandas.read_excel(args.file_name, sheet_name=["730_samples", "54_samples"]), ignore_index=True)
data = data[["관리번호", "Classification", "AL", "PD", "DNA농도(ng/ul)", "Total bacteria"] + using_features]

if not args.include_ap:
    data = data.loc[~(data["Classification"] == "AP")]
    data.reset_index(inplace=True)

data.rename(columns={"관리번호": "Number", "DNA농도(ng/ul)": "DNA"}, inplace=True)
data["Classification"] = list(map(lambda x: {"AP": "S", "Healthy": "H", "CP_E": "E", "CP_M": "M", "CP_S": "S"}[x], data["Classification"]))
data["Classification_number"] = list(map(lambda x: {"H": 0, "E": 1, "M": 2, "S": 3}[x], data["Classification"]))

if args.verbose:
    print("Train data:")
    print(data)

if not os.path.exists(args.pickle_dir):
    if args.verbose:
        print("Making pickle directory as:", args.pickle_dir)
    os.makedirs(args.pickle_dir)
elif os.path.isdir(args.pickle_dir):
    if args.verbose:
        print("Pickle directory already exists")
elif os.path.isfile(args.pickle_dir):
    raise ValueError("This is a file: " + args.pickle_dir)
else:
    exit("Something went wrong")

with open(os.path.join(args.pickle_dir, "command.sh"), "w") as f:
    f.write("python3 ")
    f.write(" ".join(sys.argv))

args.png_dir = os.path.join(args.pickle_dir, args.png_dir)
if not os.path.exists(args.png_dir):
    if args.verbose:
        print("Making PNG directory as:", args.png_dir)
    os.makedirs(args.png_dir)
elif os.path.isdir(args.png_dir):
    if args.verbose:
        print("PNG directory already exists")
elif os.path.isfile(args.png_dir):
    raise ValueError("This is a file: " + args.png_dir)
else:
    exit("Something went wrong")

if args.verbose:
    print("Generating TSNE")
tsne_pickle = os.path.join(args.pickle_dir, "tsne.csv")
if os.path.exists(tsne_pickle) and args.remake > remake_where["TSNE"]:
    if args.verbose:
        print("Pickle exists on TSNE")

    tsne_data = pandas.read_csv(tsne_pickle)
else:
    if not os.path.exists(tsne_pickle):
        if args.verbose:
            print("There is no pickle file for TSNE")
    elif args.remake > remake_where["TSNE"]:
        print("Pickle file will be overwritten")

    tmp_data = data[["Total bacteria"] + using_features]

    tsne_data = pandas.DataFrame(sklearn.manifold.TSNE(n_components=2, random_state=0).fit_transform(tmp_data), columns=["TSNE1", "TSNE2"])
    for column in list(tsne_data.columns):
        tsne_data[column] = scipy.stats.zscore(tsne_data[column])
    for column in ["Number"]:
        tsne_data[column] = data[column]

    tsne_data.to_csv(tsne_pickle, index=False)
if args.verbose:
    print(tsne_data)
    print("Done!!")

if args.verbose:
    print("Drawing TSNE Plot")
matplotlib.use("Agg")
matplotlib.rcParams.update({"font.size": 30})

matplotlib.pyplot.figure()
for color, item in zip(["g", "b", "r", "k"], ["H", "E", "M", "S"]):
    selected_data = tsne_data.loc[(data["Classification"] == item)]
    matplotlib.pyplot.scatter(selected_data["TSNE1"], selected_data["TSNE2"], c=color, marker="$%s$" % item, s=200, label=item)

matplotlib.pyplot.xlabel("TSNE-1")
matplotlib.pyplot.ylabel("TSNE-2")
matplotlib.pyplot.grid(True)
matplotlib.pyplot.legend()

fig = matplotlib.pyplot.gcf()
fig.set_size_inches(24, 24)
fig.savefig(os.path.join(args.png_dir, "TSNE.png"))
matplotlib.pyplot.close()
if args.verbose:
    print("Done!!")

train_data, validation_data = sklearn.model_selection.train_test_split(data[["Classification", "Classification_number"] + using_features], test_size=0.1, random_state=args.random_state)
if args.verbose:
    print(train_data)
    print(validation_data)


def num_to_bacteria(num, bacteria):
    if num >= 2 ** len(bacteria):
        raise IndexError
    return list(map(lambda x: x[1], list(filter(lambda x: num & (2 ** x[0]), list(enumerate(bacteria))))))


def aggregate_confusion_matrix(confusion_matrix):
    TP, FP, FN, TN = confusion_matrix[0][0], confusion_matrix[0][1], confusion_matrix[1][0], confusion_matrix[1][1]
    return (TP / (TP + FN), TN / (TN + FP), TP / (TP + FP), TN / (TN + FN), FN / (FN + TP), FP / (FP + TN), FP / (FP + TP), FN / (FN + TN), TP / (TP + FN + FP), (TP + TN) / (TP + TN + FP + FN), 2 * TP / (2 * TP + FP + FN), (TP / FP) / (FN / TN))


def run_four_group_classification(num, features, classifier):
    classifier.fit(train_data[features], numpy.ravel(train_data[["Classification"]]))

    prediction = classifier.predict(validation_data[features])

    return (num, sklearn.metrics.roc_auc_score(validation_data[["Classification_number"]], classifier.predict_proba(validation_data[features]), multi_class="ovr"), sklearn.metrics.balanced_accuracy_score(validation_data[["Classification"]], prediction)) + aggregate_confusion_matrix(numpy.sum(sklearn.metrics.multilabel_confusion_matrix(validation_data[["Classification"]], prediction), axis=0, dtype=int))


fourgroup_classifier_results = list()
for classifier_name, classifier in multiclass_classifier_list:
    if args.verbose:
        print(classifier_name)
    csv_file = os.path.join(args.pickle_dir, "FourGroups_" + classifier_name + ".csv")
    if os.path.isfile(csv_file) and args.remake > remake_where["four_groups"]:
        classifier_result = pandas.read_csv(csv_file)
    else:
        classifier_result = [("Number", "area_under_curve", "balanced_acuuracy") + statistics]
        with multiprocessing.Pool(processes=args.jobs) as pool:
            classifier_result += sorted(pool.starmap(run_four_group_classification, [(i, num_to_bacteria(i, using_features), classifier) for i in range(1, 2 ** len(using_features))]))

        classifier_result = pandas.DataFrame(classifier_result[1:], columns=classifier_result[0])
        classifier_result.to_csv(csv_file, index=False)

    if args.verbose:
        print(classifier_result)
    fourgroup_classifier_results.append(classifier_result)
