import argparse
import matplotlib
import matplotlib.pyplot
import pandas
import seaborn
import statannot
import general


def draw_scatter(input_file, output_file):
    seaborn.set(context="poster", style="whitegrid")

    fig, ax = matplotlib.pyplot.subplots(figsize=(24, 24))
    seaborn.scatterplot(data=pandas.read_csv(input_file), x="AL", y="PD", hue="Classification", style="Classification", legend="full", ax=ax, hue_order=general.classes)

    fig.savefig(general.check_exist(output_file))
    matplotlib.pyplot.close(fig)


def draw_violin(input_file, output_file, watch):
    data = pandas.read_csv(input_file)

    seaborn.set(context="poster", style="whitegrid")

    fig, ax = matplotlib.pyplot.subplots(figsize=(24, 24))
    seaborn.violinplot(data=data, x="Classification", y=watch, order=general.classes)
    statannot.add_stat_annotation(ax, data=data, x="Classification", y=watch, box_pairs=[(general.classes[i - 1], general.classes[i]) for i in range(1, len(general.classes))], test="t-test_ind", text_format="star", verbose=0, order=general.classes)

    fig.savefig(general.check_exist(output_file))
    matplotlib.pyplot.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    group1 = parser.add_mutually_exclusive_group(required=True)
    group1.add_argument("--scatter", help="Scatter plot", action="store_true", default=False)
    group1.add_argument("--violin", help="Violin plot", action="store_true", default=False)

    parser.add_argument("-i", "--input_file", help="File name to input", default=None)
    parser.add_argument("-o", "--output_file", help="File name to output", default=None)
    parser.add_argument("-c", "--column", help="Column to draw plot", choices=["AL", "PD"], default="AL")

    args = parser.parse_args()

    if args.scatter:
        draw_scatter(args.input_file, args.output_file)
    elif args.violin:
        draw_violin(args.input_file, args.output_file, args.column)
    else:
        exit("Something went wrong")
