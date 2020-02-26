import argparse
import os
import matplotlib
import matplotlib.pyplot
import seaborn
import pandas
import general


def draw_statistics(csv_file, output_dir):
    if not os.path.isfile(csv_file):
        raise ValueError(csv_file)

    statistics_data = pandas.read_csv(csv_file)
    statistics_data["feature_num"] = list(map(lambda x: len(general.num_to_bacteria(x)), statistics_data["Number"]))

    for combined_class in sorted(set(statistics_data["combined_class"])):
        selected_data = statistics_data.loc[(statistics_data["combined_class"] == combined_class)]

        for statistics_value in sorted(general.aggregate_confusion_matrix(None)):
            seaborn.set(context="poster", style="whitegrid")
            fig, ax = matplotlib.pyplot.subplots(figsize=(24, 24))
            seaborn.lineplot(x="feature_num", y=statistics_value, hue="classifier", ax=ax, legend="full", data=selected_data, hue_order=sorted(set(statistics_data["classifier"])), estimator="median", ci="sd")
            ax.set_title(combined_class.replace("-", " "))
            fig.savefig(general.check_exist(os.path.join(output_dir, "Median_" + combined_class + "_" + statistics_value + ".png")))
            matplotlib.pyplot.close(fig)

            seaborn.set(context="poster", style="whitegrid")
            fig, ax = matplotlib.pyplot.subplots(figsize=(24, 24))
            seaborn.lineplot(x="feature_num", y=statistics_value, hue="classifier", ax=ax, legend="full", data=selected_data, hue_order=sorted(set(statistics_data["classifier"])))
            ax.set_title(combined_class.replace("-", " "))
            fig.savefig(general.check_exist(os.path.join(output_dir, "Mean_" + combined_class + "_" + statistics_value + ".png")))
            matplotlib.pyplot.close(fig)

            seaborn.set(context="poster", style="whitegrid")
            fig, ax = matplotlib.pyplot.subplots(figsize=(24, 24))
            seaborn.lineplot(x="feature_num", y=statistics_value, hue="classifier", ax=ax, legend="full", data=selected_data, hue_order=sorted(set(statistics_data["classifier"])), estimator=min, ci=None)
            ax.set_title(combined_class.replace("-", " "))
            fig.savefig(general.check_exist(os.path.join(output_dir, "Min_" + combined_class + "_" + statistics_value + ".png")))
            matplotlib.pyplot.close(fig)

            seaborn.set(context="poster", style="whitegrid")
            fig, ax = matplotlib.pyplot.subplots(figsize=(24, 24))
            seaborn.lineplot(x="feature_num", y=statistics_value, hue="classifier", ax=ax, legend="full", data=selected_data, hue_order=sorted(set(statistics_data["classifier"])), estimator=max, ci=None)
            ax.set_title(combined_class.replace("-", " "))
            fig.savefig(general.check_exist(os.path.join(output_dir, "Max_" + combined_class + "_" + statistics_value + ".png")))
            matplotlib.pyplot.close(fig)


def draw_extreme(csv_file, output_dir):
    if not os.path.isfile(csv_file):
        raise ValueError(csv_file)

    statistics_data = pandas.read_csv(csv_file)

    results = [("combined_class", "classifier", "bacteria", "statistics", "type", "value")]
    for combined_class in sorted(set(statistics_data["combined_class"])):
        tmp = list(filter(lambda x: "+" in x, combined_class.split("-vs-")))
        if tmp:
            combined_class_num = general.class_to_num(tmp[0].split("+"))
        else:
            combined_class_num = 0

        for classifier in sorted(set(statistics_data["classifier"])):
            prediction_directory = os.path.join(os.path.dirname(csv_file), classifier)

            for statistics_value in general.aggregate_confusion_matrix(None):
                selected_data = statistics_data.loc[(statistics_data["combined_class"] == combined_class) & (statistics_data["classifier"] == classifier)][[statistics_value, "Number"]]

                minimum, maximum = selected_data.loc[selected_data.idxmin(axis="index")[statistics_value], "Number"], selected_data.loc[selected_data.idxmax(axis="index")[statistics_value], "Number"]

                for name, value in zip(["minimum", "maximum"], [minimum, maximum]):
                    if combined_class_num:
                        prediction_data = pandas.read_csv(os.path.join(prediction_directory, "Prediction_%s_%d.csv" % (value, combined_class_num)))
                    else:
                        prediction_data = pandas.read_csv(os.path.join(prediction_directory, "Prediction_%s.csv" % (value)))
                    prediction_data = prediction_data.groupby(list(prediction_data.columns), as_index=False).size().reset_index().rename(columns={0: "counts"}).pivot("prediction", "real", "counts").fillna(0)

                    seaborn.set(context="poster", style="whitegrid")
                    fig, ax = matplotlib.pyplot.subplots(figsize=(24, 24))
                    seaborn.heatmap(prediction_data, annot=True, ax=ax, robust=True)
                    ax.set_title(combined_class.replace("-", " ") + " with " + statistics_value)
                    fig.savefig(general.check_exist(os.path.join(output_dir, name + "_" + combined_class + "_" + classifier + "_" + statistics_value + ".png")))
                    matplotlib.pyplot.close(fig)

                    results.append((combined_class, classifier, "+".join(general.num_to_bacteria(value)), statistics_value, name, value))

    pandas.DataFrame(results[1:], columns=results[0]).to_csv(general.check_exist(os.path.join(output_dir, "Min_Max.csv")), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-v", "--version", help="Verbose output", action="store_true", default=False)
    parser.add_argument("-i", "--input_file", help="Input file", default=None)
    parser.add_argument("-o", "--output_dir", help="Output file", default=None)

    group1 = parser.add_mutually_exclusive_group(required=True)
    group1.add_argument("--stat", help="Draw statistics values", action="store_true", default=False)
    group1.add_argument("--extreme", help="Draw extreme values", action="store_true", default=False)

    args = parser.parse_args()

    if args.stat:
        draw_statistics(args.input_file, args.output_dir)
    elif args.extreme:
        draw_extreme(args.input_file, args.output_dir)
    else:
        exit("Something went wrong")
