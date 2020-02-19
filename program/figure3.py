import argparse
import os
import pandas
import seaborn

parser = argparse.ArgumentParser()

parser.add_argument("-v", "--verbose", help="Verbose output", action="store_true", default=False)
parser.add_argument("-f", "--file_name", help="File name to read data (should be XLSX)", type=str, default="/BiO/Store/Helixco/Periodontist_Fall2019/data/Periodontitis_input_dataset_from_784samples_and_additional_54samples_20190730.xlsx")
parser.add_argument("-b", "--bacteria", help="Bacteria to use (e.g. Aa)", action="append", type=str, default=[], choices=sorted(["Aa", "Pg", "Tf", "Td", "Pi", "Fn", "Pa", "Cr", "Ec"]))
parser.add_argument("-c", "--classification", help="Classification to use (amongst Healthy, CP_E, CP_M & CP_S)", action="append", type=str, default=[], choices=["Healthy", "CP_E", "CP_M", "CP_S"])
parser.add_argument("--include_ap", help="Include AP", action="store_true", default=False)
parser.add_argument("-p", "--png", help="PNG file name", type=str, default="pair.png")
parser.add_argument("-m", "--method", help="Method to calculate correlation", choices=["pearson", "kendall", "spearman"], default="pearson")

group1 = parser.add_mutually_exclusive_group(required=True)
group1.add_argument("--abs", help="Use absolute values", action="store_true", default=False)
group1.add_argument("--rel", help="Use relative values", action="store_true", default=False)
group1.add_argument("--both", help="Use both absolute and relative", action="store_true", default=False)

args = parser.parse_args()

if not os.path.exists(args.file_name) or not os.path.isfile(args.file_name):
    exit("Invalid file: " + args.file_name)
if not args.file_name.endswith(".xlsx"):
    exit("Invalid file: " + args.file_name)
if not args.png.endswith(".png"):
    exit("Invalid file: " + args.png)

data = pandas.concat(pandas.read_excel(args.file_name, sheet_name=["730_samples", "54_samples"]), ignore_index=True)

if not args.bacteria:
    if args.verbose:
        print("Use default Bacteria")
    args.bacteria = sorted(["Aa", "Pg", "Tf", "Td", "Pi", "Fn", "Pa", "Cr", "Ec"])

if not args.classification:
    if args.verbose:
        print("Use default Classification")
    args.classification = ["Healthy", "CP_E", "CP_M", "CP_S"]

if not args.include_ap:
    data = data.loc[~(data["Classification"] == "AP")]

if args.abs:
    pass
elif args.rel:
    args.bacteria = [x + "_relative" for x in args.bacteria]
elif args.both:
    args.bacteria += [x + "_relative" for x in args.bacteria]
else:
    exit("Something went wrong")

seaborn.set(context="poster", style="whitegrid")
seaborn.pairplot(data, vars=args.bacteria, hue="Classification", kind="reg", diag_kind="kde", markers="+").savefig(args.png)
