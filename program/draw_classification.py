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

            seaborn.lineplot(x="feature_num", y=statistics_value, hue="classifier", ax=ax, legend="full", data=selected_data)

            ax.set_title(combined_class.replace("-", " "))

            fig.savefig(general.check_exist(os.path.join(output_dir, combined_class + "_" + statistics_value + ".png")))
            matplotlib.pyplot.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-v", "--version", help="Verbose output", action="store_true", default=False)
    parser.add_argument("-i", "--input_file", help="Input file", default=None)
    parser.add_argument("-o", "--output_dir", help="Output file", default=None)

    group1 = parser.add_mutually_exclusive_group(required=True)
    group1.add_argument("--stat", help="Draw statistics values", action="store_true", default=False)

    args = parser.parse_args()

    if args.stat:
        draw_statistics(args.input_file, args.output_dir)
    else:
        exit("Something went wrong")
