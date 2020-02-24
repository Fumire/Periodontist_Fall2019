import argparse
import os
import multiprocessing
import matplotlib
import matplotlib.pyplot
import numpy
import seaborn
import sklearn.ensemble
import sklearn.gaussian_process
import sklearn.metrics
import sklearn.model_selection
import sklearn.neighbors
import sklearn.neural_network
import sklearn.svm
import sklearn.tree
import pandas
import general

classifiers = [("KNeighbors", sklearn.neighbors.KNeighborsClassifier(algorithm="brute", n_jobs=1)), ("SVC", sklearn.svm.SVC(probability=True, decision_function_shape="ovr", random_state=0)), ("Gaussian", sklearn.gaussian_process.GaussianProcessClassifier(max_iter_predict=2 ** 30, random_state=0, multi_class="one_vs_rest", n_jobs=1)), ("DecisionTree", sklearn.tree.DecisionTreeClassifier(random_state=0)), ("RandomeForest", sklearn.ensemble.RandomForestClassifier(random_state=0, n_jobs=1, class_weight="balanced")), ("NeuralNetwork", sklearn.neural_network.MLPClassifier(max_iter=2 ** 30, random_state=0, early_stopping=True)), ("AdaBoost", sklearn.ensemble.AdaBoostClassifier(random_state=0))]


def actual_four_class_classifier(classifier, train_data, test_data, output_dir, bacteria_num):
    train_answer = train_data.pop("Classification")
    test_answer = test_data.pop("Classification")

    train_data = train_data[general.num_to_bacteria(bacteria_num)]
    test_data = test_data[general.num_to_bacteria(bacteria_num)]

    classifier.fit(train_data, train_answer)

    pandas.DataFrame(classifier.predict_proba(test_data), columns=general.classes).to_csv(general.check_exist(os.path.join(output_dir, str(bacteria_num) + ".csv")), index=False)

    prediction = classifier.predict(test_data)
    return (bacteria_num, sklearn.metrics.balanced_accuracy_score(test_answer, prediction)) + general.aggregate_confusion_matrix(numpy.sum(sklearn.metrics.multilabel_confusion_matrix(test_answer, prediction), axis=0, dtype=int))


def headquarter_four_class_classifier(jobs=30, input_file=None, output_dir=None):
    if (input_file is None) or (output_dir is None):
        raise ValueError
    elif not os.path.isfile(input_file):
        raise ValueError(input_file)

    data = pandas.read_csv(input_file)
    data = data[["Classification"] + general.whole_values]

    train_data, test_data = sklearn.model_selection.train_test_split(data, test_size=0.1, random_state=0, stratify=data[["Classification"]])

    with multiprocessing.Pool(processes=jobs) as pool:
        for name, classifier in classifiers:
            results = [("Number", "balanced_accuracy_score") + general.aggregate_confusion_matrix(None)]

            results += pool.starmap(actual_four_class_classifier, [(classifier, train_data.copy(), test_data.copy(), os.path.join(output_dir, name), i) for i in range(1, 2 ** len(general.absolute_values))])
            results += pool.starmap(actual_four_class_classifier, [(classifier, train_data.copy(), test_data.copy(), os.path.join(output_dir, name), i * (2 ** len(general.absolute_values))) for i in range(1, 2 ** len(general.relative_values))])

            pandas.DataFrame(results[1:], columns=results[0]).to_csv(general.check_exist(os.path.join(output_dir, name, "statistics.csv")), index=False)

            data = pandas.read_csv(os.path.join(output_dir, name, "statistics.csv"))
            data["Bacteria_Num"] = list(map(lambda x: len(general.num_to_bacteria(x)), data["Number"]))

            for value in ("balanced_accuracy_score", ) + general.aggregate_confusion_matrix(None):
                seaborn.set(context="poster", style="whitegrid")

                fig, ax = matplotlib.pyplot.subplots(figsize=(24, 24))
                seaborn.lineplot(x="Bacteria_Num", y=value, data=data, ax=ax)

                ax.set_title("4-class with " + name)

                fig.savefig(general.check_exist(os.path.join(output_dir, name, value + ".png")))
                matplotlib.pyplot.close(fig)


def actual_three_class_classifier(classifier, train_data, test_data, output_dir, bacteria_num):
    train_answer = train_data.pop("Classification")
    test_answer = test_data.pop("Classification")

    train_data = train_data[general.num_to_bacteria(bacteria_num)]
    test_data = test_data[general.num_to_bacteria(bacteria_num)]

    classifier.fit(train_data, train_answer)

    prediction = classifier.predict(test_data)
    return (bacteria_num, sklearn.metrics.balanced_accuracy_score(test_answer, prediction)) + general.aggregate_confusion_matrix(numpy.sum(sklearn.metrics.multilabel_confusion_matrix(test_answer, prediction), axis=0, dtype=int))


def headquarter_three_class_classifier(jobs=30, input_file=None, output_dir=None):
    if (input_file is None) or (output_dir is None):
        raise ValueError
    elif not os.path.isfile(input_file):
        raise ValueError(input_file)

    data = pandas.read_csv(input_file)
    data = data[["Classification"] + general.whole_values]

    for one_class, two_class in general.two_class_combinations:
        class_column = list(map(lambda x: one_class + two_class if x in [one_class, two_class] else x, list(data["Classification"])))
        train_data, test_data = sklearn.model_selection.train_test_split(data, test_size=0.1, random_state=0, stratify=class_column)

        with multiprocessing.Pool(processes=jobs) as pool:
            for name, classifier in classifiers:
                results = [("Number", "balanced_accuracy_score") + general.aggregate_confusion_matrix(None)]

                results += pool.starmap(actual_three_class_classifier, [(classifier, train_data.copy(), test_data.copy(), os.path.join(output_dir, name), i) for i in range(1, 2 ** len(general.absolute_values))])
                results += pool.starmap(actual_three_class_classifier, [(classifier, train_data.copy(), test_data.copy(), os.path.join(output_dir, name), i * (2 ** len(general.absolute_values))) for i in range(1, 2 ** len(general.relative_values))])

                pandas.DataFrame(results[1:], columns=results[0]).to_csv(general.check_exist(os.path.join(output_dir, name, one_class + "-" + two_class + ".csv")), index=False)

                data = pandas.read_csv(os.path.join(output_dir, name, one_class + "-" + two_class + ".csv"))
                data["Bacteria_Num"] = list(map(lambda x: len(general.num_to_bacteria(x)), data["Number"]))

                for value in ("balanced_accuracy_score", ) + general.aggregate_confusion_matrix(None):
                    seaborn.set(context="poster", style="whitegrid")

                    fig, ax = matplotlib.pyplot.subplots(figsize=(24, 24))
                    seaborn.lineplot(x="Bacteria_Num", y=value, data=data, ax=ax)

                    ax.set_title("3-class (%s+%s) with " % (one_class, two_class) + name)

                    fig.savefig(general.check_exist(os.path.join(output_dir, name, one_class + "-" + two_class + "-" + value + ".png")))
                    matplotlib.pyplot.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    group1 = parser.add_mutually_exclusive_group(required=True)
    group1.add_argument("--four", help="Use four-class classification", action="store_true", default=False)
    group1.add_argument("--three", help="Use three-class classification", action="store_true", default=False)

    parser.add_argument("-v", "--verbose", help="Verbose output", action="store_true", default=False)
    parser.add_argument("-i", "--input_file", help="File name to input", default=None)
    parser.add_argument("-o", "--output_dir", help="Directory name to output", default=None)
    parser.add_argument("-j", "--jobs", help="Number of parallel jobs", type=int, default=30)

    args = parser.parse_args()

    if args.four:
        headquarter_four_class_classifier(jobs=args.jobs, input_file=args.input_file, output_dir=args.output_dir)
    elif args.three:
        headquarter_three_class_classifier(jobs=args.jobs, input_file=args.input_file, output_dir=args.output_dir)
    else:
        exit("Something went wrong")
