import argparse
import multiprocessing
import os
import matplotlib
import matplotlib.pyplot
import seaborn
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.svm
import pandas
import general

max_iteration = 100
regressors = [("LinearRegression", sklearn.linear_model.LinearRegression(n_jobs=1)), ("Ridge", sklearn.linear_model.Ridge(max_iter=max_iteration, random_state=0)), ("SVR", sklearn.svm.SVR(max_iter=max_iteration)), ("NuSVR", sklearn.svm.NuSVR(max_iter=max_iteration)), ("LinearSVR", sklearn.svm.LinearSVR(random_state=0, max_iter=max_iteration))]


def actual_regressor(regressor, train_data, test_data, output_dir, bacteria_num):
    train_answer = train_data.pop("answer")
    test_answer = test_data.pop("answer")

    train_data = train_data[general.num_to_bacteria(bacteria_num)]
    test_data = test_data[general.num_to_bacteria(bacteria_num)]

    regressor.fit(train_data, train_answer)

    return bacteria_num, regressor.score(test_data, test_answer)


def headquarter_regressor(input_file, output_dir, watch, jobs=30):
    data = pandas.read_csv(input_file)
    data = data[[watch] + general.whole_values]
    data.rename(columns={watch: "answer"}, inplace=True)

    train_data, test_data = sklearn.model_selection.train_test_split(data, test_size=0.1, random_state=0)

    with multiprocessing.Pool(processes=jobs) as pool:
        for name, regressor in regressors:
            results = [("Number", "R2_score")]

            results += pool.starmap(actual_regressor, [(regressor, train_data.copy(), test_data.copy(), os.path.join(output_dir, name), i) for i in range(1, 2 ** len(general.absolute_values))])
            results += pool.starmap(actual_regressor, [(regressor, train_data.copy(), test_data.copy(), os.path.join(output_dir, name), i * (2 ** len(general.absolute_values))) for i in range(1, 2 ** len(general.relative_values))])

            results = pandas.DataFrame(results[1:], columns=results[0])
            results["regressor"] = name
            results["feature_num"] = list(map(lambda x: len(general.num_to_bacteria(x)), results["Number"]))
            results.to_csv(general.check_exist(os.path.join(output_dir, name, "statistics.csv")), index=False)

    drawing_data = pandas.concat([pandas.read_csv(os.path.join(output_dir, name, "statistics.csv")) for name, regressor in regressors], ignore_index=True)
    drawing_data.to_csv(general.check_exist(os.path.join(output_dir, "statistics.csv")), index=False)

    seaborn.set(context="poster", style="whitegrid")
    fig, ax = matplotlib.pyplot.subplots(figsize=(24, 24))

    seaborn.lineplot(data=drawing_data, x="feature_num", y="R2_score", hue="regressor", ax=ax, legend="full", hue_order=sorted(set(drawing_data["regressor"])))

    fig.savefig(general.check_exist(os.path.join(output_dir, "Regressor_" + watch + ".png")))
    matplotlib.pyplot.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_file", help="File name to input", required=True)
    parser.add_argument("-o", "--output_dir", help="Directory name to output", required=True)
    parser.add_argument("-j", "--jobs", help="Number of processes", type=int, default=30)
    parser.add_argument("-c", "--column", help="Column to predict", choices=["AL", "PD"], default="AL")

    args = parser.parse_args()

    headquarter_regressor(args.input_file, args.output_dir, args.column, args.jobs)
