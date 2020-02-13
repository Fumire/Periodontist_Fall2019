import argparse
import os
import matplotlib
import matplotlib.pyplot
import pandas
import scipy
import sklearn.manifold

parser = argparse.ArgumentParser()

parser.add_argument("-v", "--verbose", help="Verbose output", action="store_true", default=False)
parser.add_argument("-f", "--file_name", help="File name to read data", type=str, default="/BiO/Store/Helixco/Periodontist_Fall2019/data/Periodontitis_input_dataset_from_784samples_and_additional_54samples_20190730.xlsx")
parser.add_argument("-b", "--bacteria", help="Bacteria to use", action="append", type=str, default=[])
parser.add_argument("-c", "--classification", help="Classification to use", action="append", type=str, default=[])
parser.add_argument("--include_ap", help="Include AP", action="store_true", default=False)
parser.add_argument("-p", "--png", help="PNG file name", type=str, default="TSNE.png")

group1 = parser.add_mutually_exclusive_group(required=True)
group1.add_argument("--abs", help="Use absolute values", action="store_true", default=False)
group1.add_argument("--rel", help="Use relative values", action="store_true", default=False)
group1.add_argument("--both", help="Use both absolute and relative", action="store_true", default=False)

args = parser.parse_args()

if not os.path.exists(args.file_name) or not os.path.isfile(args.file_name):
    exit("Invalid file: " + args.file_name)

data = pandas.concat(pandas.read_excel(args.file_name, sheet_name=["730_samples", "54_samples"]), ignore_index=True)

if not args.bacteria:
    if args.verbose:
        print("Use default Bacteria")
    args.bacteria = ["Aa", "Pg", "Tf", "Td", "Pi", "Fn", "Pa", "Cr", "Ec"]

if not args.classification:
    if args.verbose:
        print("Use default Classification")
    args.classification = ["Healthy", "CP_E", "CP_M", "CP_S"]
if args.include_ap:
    args.classification += ["AP"]

if args.abs:
    pass
elif args.rel:
    args.bacteria = [x + "_relative" for x in args.bacteria]
elif args.both:
    args.bacteria += [x + "_relative" for x in args.bacteria]
else:
    exit("Something went wrong")

data = data[["Classification"] + args.bacteria]
if args.verbose:
    print(data)

tsne_data = pandas.DataFrame(sklearn.manifold.TSNE(n_components=2, random_state=0).fit_transform(data[args.bacteria]), columns=["TSNE1", "TSNE2"])
for column in list(tsne_data.columns):
    tsne_data[column] = scipy.stats.zscore(tsne_data[column])
if args.verbose:
    print(tsne_data)

matplotlib.use("Agg")
matplotlib.rcParams.update({"font.size": 30})

matplotlib.pyplot.figure()
for item in args.classification:
    selected_data = tsne_data.loc[(data["Classification"] == item)]
    color = {"Healthy": "g", "CP_E": "b", "CP_M": "r", "CP_S": "k", "AP": "k"}[item]
    marker = {"Healthy": "$H$", "CP_E": "$E$", "CP_M": "$M$", "CP_S": "$S$", "AP": "$S$"}[item]
    label = {"Healthy": "Healthy", "CP_E": "Early", "CP_M": "Moderate", "CP_S": "Severe", "AP": "Severe"}[item]
    matplotlib.pyplot.scatter(selected_data["TSNE1"], selected_data["TSNE2"], c=color, marker=marker, s=200, label=label)

matplotlib.pyplot.xlabel("TSNE-1")
matplotlib.pyplot.ylabel("TSNE-2")
matplotlib.pyplot.grid(True)
matplotlib.pyplot.legend()

fig = matplotlib.pyplot.gcf()
fig.set_size_inches(24, 24)
fig.savefig(args.png)
matplotlib.pyplot.close()
