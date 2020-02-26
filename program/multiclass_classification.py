import argparse
import os
import multiprocessing
import numpy
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

max_iteration = 2 ** 30
classifiers = [("KNeighbors", sklearn.neighbors.KNeighborsClassifier(algorithm="brute", n_jobs=1)), ("LinearSVC", sklearn.svm.SVC(kernel="linear", probability=True, decision_function_shape="ovr", random_state=0, class_weight="balanced", max_iter=max_iteration)), ("PolySVC", sklearn.svm.SVC(kernel="poly", probability=True, decision_function_shape="ovr", random_state=0, class_weight="balanced", max_iter=max_iteration)), ("RbfSVC", sklearn.svm.SVC(kernel="rbf", probability=True, decision_function_shape="ovr", random_state=0, class_weight="balanced", max_iter=max_iteration)), ("sigmoidSVC", sklearn.svm.SVC(kernel="sigmoid", probability=True, decision_function_shape="ovr", random_state=0, class_weight="balanced", max_iter=max_iteration)), ("DecisionTree", sklearn.tree.DecisionTreeClassifier(random_state=0, class_weight="balanced")), ("RandomForest", sklearn.ensemble.RandomForestClassifier(random_state=0, n_jobs=1, class_weight="balanced")), ("AdamNN", sklearn.neural_network.MLPClassifier(max_iter=max_iteration, random_state=0, early_stopping=True, solver="adam")), ("lbfgsNN", sklearn.neural_network.MLPClassifier(max_iter=max_iteration, random_state=0, early_stopping=True, solver="lbfgs")), ("AdaBoost", sklearn.ensemble.AdaBoostClassifier(random_state=0))]


def actual_five_class_classifier(classifier, train_data, test_data, output_dir, bacteria_num):
    train_answer = train_data.pop("Classification")
    test_answer = test_data.pop("Classification")

    train_data = train_data[general.num_to_bacteria(bacteria_num)]
    test_data = test_data[general.num_to_bacteria(bacteria_num)]

    classifier.fit(train_data, train_answer)

    pandas.DataFrame(classifier.predict_proba(test_data), columns=sorted(set(test_answer))).to_csv(general.check_exist(os.path.join(output_dir, "Probability_" + str(bacteria_num) + ".csv")), index=False)

    prediction = classifier.predict(test_data)
    pandas.DataFrame(zip(test_answer, prediction), columns=["real", "prediction"]).to_csv(general.check_exist(os.path.join(output_dir, "Prediction_" + str(bacteria_num) + ".csv")), index=False)
    return (bacteria_num, sklearn.metrics.balanced_accuracy_score(test_answer, prediction)) + general.aggregate_confusion_matrix(numpy.sum(sklearn.metrics.multilabel_confusion_matrix(test_answer, prediction), axis=0, dtype=int))


def headquarter_five_class_classifier(jobs=30, input_file=None, output_dir=None):
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

            results += pool.starmap(actual_five_class_classifier, [(classifier, train_data.copy(), test_data.copy(), os.path.join(output_dir, name), i) for i in range(1, 2 ** len(general.absolute_values))])
            results += pool.starmap(actual_five_class_classifier, [(classifier, train_data.copy(), test_data.copy(), os.path.join(output_dir, name), i * (2 ** len(general.absolute_values))) for i in range(1, 2 ** len(general.relative_values))])

            results = pandas.DataFrame(results[1:], columns=results[0])
            results["classifier"] = name
            results["combined_class"] = "-vs-".join(sorted(set(data["Classification"])))
            results.to_csv(general.check_exist(os.path.join(output_dir, name, "statistics.csv")), index=False)

    pandas.concat([pandas.read_csv(os.path.join(output_dir, name, "statistics.csv")) for name, classifier in classifiers], ignore_index=True).to_csv(general.check_exist(os.path.join(output_dir, "statistics.csv")), index=False)


def actual_four_class_classifier(classifier, train_data, test_data, output_dir, bacteria_num, class_num):
    train_answer = train_data.pop("Classification")
    test_answer = test_data.pop("Classification")

    train_data = train_data[general.num_to_bacteria(bacteria_num)]
    test_data = test_data[general.num_to_bacteria(bacteria_num)]

    classifier.fit(train_data, train_answer)

    pandas.DataFrame(classifier.predict_proba(test_data), columns=sorted(set(test_answer))).to_csv(general.check_exist(os.path.join(output_dir, "Probability_" + str(bacteria_num) + "_" + str(class_num) + ".csv")), index=False)

    prediction = classifier.predict(test_data)
    pandas.DataFrame(zip(test_answer, prediction), columns=["real", "prediction"]).to_csv(general.check_exist(os.path.join(output_dir, "Prediction_" + str(bacteria_num) + "_" + str(class_num) + ".csv")), index=False)

    return (bacteria_num, sklearn.metrics.balanced_accuracy_score(test_answer, prediction)) + general.aggregate_confusion_matrix(numpy.sum(sklearn.metrics.multilabel_confusion_matrix(test_answer, prediction), axis=0, dtype=int))


def headquarter_four_class_classifier(jobs=30, input_file=None, output_dir=None):
    if (input_file is None) or (output_dir is None):
        raise ValueError
    elif not os.path.isfile(input_file):
        raise ValueError(input_file)

    data = pandas.read_csv(input_file)
    data = data[["Classification"] + general.whole_values]
    result_data = list()

    original_class = list(data["Classification"])

    for selected_class in general.two_class_combinations:
        data["Classification"] = list(map(lambda x: "+".join(selected_class) if x in selected_class else x, data["Classification"]))
        train_data, test_data = sklearn.model_selection.train_test_split(data, test_size=0.1, random_state=0, stratify=data["Classification"])

        with multiprocessing.Pool(processes=jobs) as pool:
            for name, classifier in classifiers:
                results = [("Number", "balanced_accuracy_score") + general.aggregate_confusion_matrix(None)]

                results += pool.starmap(actual_four_class_classifier, [(classifier, train_data.copy(), test_data.copy(), os.path.join(output_dir, name), i, general.class_to_num(selected_class)) for i in range(1, 2 ** len(general.absolute_values))])
                results += pool.starmap(actual_four_class_classifier, [(classifier, train_data.copy(), test_data.copy(), os.path.join(output_dir, name), i * (2 ** len(general.absolute_values)), general.class_to_num(selected_class)) for i in range(1, 2 ** len(general.relative_values))])

                results = pandas.DataFrame(results[1:], columns=results[0])
                results["classifier"] = name
                results["combined_class"] = "-vs-".join(sorted(set(data["Classification"])))
                results.to_csv(general.check_exist(os.path.join(output_dir, name, "-".join(selected_class) + ".csv")), index=False)

                result_data.append(results.copy())
        data["Classification"] = original_class

    pandas.concat(result_data, ignore_index=True).to_csv(general.check_exist(os.path.join(output_dir, "statistics.csv")), index=False)


def actual_three_class_classifier(classifier, train_data, test_data, output_dir, bacteria_num, class_num):
    train_answer = train_data.pop("Classification")
    test_answer = test_data.pop("Classification")

    train_data = train_data[general.num_to_bacteria(bacteria_num)]
    test_data = test_data[general.num_to_bacteria(bacteria_num)]

    classifier.fit(train_data, train_answer)

    pandas.DataFrame(classifier.predict_proba(test_data), columns=sorted(set(test_answer))).to_csv(general.check_exist(os.path.join(output_dir, "Probability_" + str(bacteria_num) + "_" + str(class_num) + ".csv")), index=False)

    prediction = classifier.predict(test_data)
    pandas.DataFrame(zip(test_answer, prediction), columns=["real", "prediction"]).to_csv(general.check_exist(os.path.join(output_dir, "Prediction_" + str(bacteria_num) + "_" + str(class_num) + ".csv")), index=False)
    return (bacteria_num, sklearn.metrics.balanced_accuracy_score(test_answer, prediction)) + general.aggregate_confusion_matrix(numpy.sum(sklearn.metrics.multilabel_confusion_matrix(test_answer, prediction), axis=0, dtype=int))


def headquarter_three_class_classifier(jobs=30, input_file=None, output_dir=None):
    if (input_file is None) or (output_dir is None):
        raise ValueError
    elif not os.path.isfile(input_file):
        raise ValueError(input_file)

    data = pandas.read_csv(input_file)
    data = data[["Classification"] + general.whole_values]
    result_data = list()

    original_class = list(data["Classification"])

    for selected_class in general.three_class_combinations:
        data["Classification"] = list(map(lambda x: "+".join(selected_class) if x in selected_class else x, data["Classification"]))
        train_data, test_data = sklearn.model_selection.train_test_split(data, test_size=0.1, random_state=0, stratify=data["Classification"])

        with multiprocessing.Pool(processes=jobs) as pool:
            for name, classifier in classifiers:
                results = [("Number", "balanced_accuracy_score") + general.aggregate_confusion_matrix(None)]

                results += pool.starmap(actual_three_class_classifier, [(classifier, train_data.copy(), test_data.copy(), os.path.join(output_dir, name), i, general.class_to_num(selected_class)) for i in range(1, 2 ** len(general.absolute_values))])
                results += pool.starmap(actual_three_class_classifier, [(classifier, train_data.copy(), test_data.copy(), os.path.join(output_dir, name), i * (2 ** len(general.absolute_values)), general.class_to_num(selected_class)) for i in range(1, 2 ** len(general.relative_values))])

                results = pandas.DataFrame(results[1:], columns=results[0])
                results["classifier"] = name
                results["combined_class"] = "-vs-".join(sorted(set(data["Classification"])))
                results.to_csv(general.check_exist(os.path.join(output_dir, name, "-".join(selected_class) + ".csv")), index=False)

                result_data.append(results.copy())

        data["Classification"] = original_class

    pandas.concat(result_data, ignore_index=True).to_csv(general.check_exist(os.path.join(output_dir, "statistics.csv")), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    group1 = parser.add_mutually_exclusive_group(required=True)
    group1.add_argument("--five", help="Use five-class classification", action="store_true", default=False)
    group1.add_argument("--four", help="Use four-class classification", action="store_true", default=False)
    group1.add_argument("--three", help="Use three-class classification", action="store_true", default=False)

    parser.add_argument("-v", "--verbose", help="Verbose output", action="store_true", default=False)
    parser.add_argument("-i", "--input_file", help="File name to input", default=None)
    parser.add_argument("-o", "--output_dir", help="Directory name to output", default=None)
    parser.add_argument("-j", "--jobs", help="Number of parallel jobs", type=int, default=30)

    args = parser.parse_args()

    if args.five:
        headquarter_five_class_classifier(jobs=args.jobs, input_file=args.input_file, output_dir=args.output_dir)
    elif args.four:
        headquarter_four_class_classifier(jobs=args.jobs, input_file=args.input_file, output_dir=args.output_dir)
    elif args.three:
        headquarter_three_class_classifier(jobs=args.jobs, input_file=args.input_file, output_dir=args.output_dir)
    else:
        exit("Something went wrong")
