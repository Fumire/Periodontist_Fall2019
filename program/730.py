import argparse
import os
import pickle
import sys
import matplotlib
import matplotlib.pyplot
import pandas
import scipy
import sklearn.manifold

parser = argparse.ArgumentParser()

parser.add_argument("--verbose", help="Verbose output", action="store_true", default=False)
parser.add_argument("--file_name", help="File name to read data", type=str, default="/BiO/Store/Helixco/Periodontist_Fall2019/data/Periodontitis_input_dataset_from_784samples_and_additional_54samples_20190730.xlsx")
parser.add_argument("--include_ap", help="Include AP / does not", action="store_true", default=False)
parser.add_argument("--pickle_dir", help="Directory to store pickle data", type=str, default="pickle")
parser.add_argument("--png_dir", help="Directory to store PNG data", type=str, default="PNG")
parser.add_argument("--tsne", help="Whether overwrite TSNE", action="store_false", default=True)

args = parser.parse_args()

if len(sys.argv) == 1:
    pass

if not os.path.exists(args.file_name) or not os.path.isfile(args.file_name):
    exit("Invalid file: " + args.file)

data = pandas.concat(pandas.read_excel(args.file_name, sheet_name=None), ignore_index=True)

data = data[["관리번호", "Classification", "AL", "PD", "DNA농도(ng/ul)", "Total bacteria", "Aa", "Pg", "Tf", "Td", "Pi", "Fn", "Pa", "Cr", "Ec", "Aa_relative", "Pg_relative", "Tf_relative", "Td_relative", "Pi_relative", "Fn_relative", "Pa_relative", "Cr_relative", "Ec_relative"]]

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
    os.mkdir(args.pickle_dir)
elif os.path.isdir(args.pickle_dir):
    if args.verbose:
        print("Pickle directory already exists")
elif os.path.isfile(args.pickle_dir):
    raise ValueError("This is a file: " + args.pickle_dir)
else:
    exit("Something went wrong")

if not os.path.exists(args.png_dir):
    if args.verbose:
        print("Making PNG directory as: ", args.png_dir)
    os.mkdir(args.png_dir)
elif os.path.isdir(args.png_dir):
    if args.verbose:
        print("PNG directory already exists")
elif os.path.isfile(args.png_dir):
    raise ValueError("This is a file: " + args.png_dir)
else:
    exit("Something went wrong")

if args.verbose:
    print("Generating TSNE")
tsne_data = pandas.DataFrame()
tsne_pickle = os.path.join(args.pickle_dir, "tsne.csv")
if os.path.exists(tsne_pickle) and args.tsne:
    if args.verbose:
        print("Pickle exists on TSNE")

    tsne_data = pandas.read_csv(tsne_pickle)
else:
    if not os.path.exists(tsne_pickle):
        print("There is no pickle file for TSNE")
    elif not args.tsne:
        print("Pickle file will be overwritten")

    tmp_data = data[["Total bacteria", "Aa", "Pg", "Tf", "Td", "Pi", "Fn", "Pa", "Cr", "Ec", "Aa_relative", "Pg_relative", "Tf_relative", "Td_relative", "Pi_relative", "Fn_relative", "Pa_relative", "Cr_relative", "Ec_relative"]]

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
