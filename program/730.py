import argparse
import os
import pickle
import sys
import pandas
import sklearn.manifold

parser = argparse.ArgumentParser()

parser.add_argument("--verbose", help="Verbose output", action="store_true", default=False)
parser.add_argument("--file_name", help="File name to read data", type=str, default="/BiO/Store/Helixco/Periodontist_Fall2019/data/Periodontitis_input_dataset_from_784samples_and_additional_54samples_20190730.xlsx")
parser.add_argument("--train_sheet", help="Sheet name to train", type=str, default="730_samples")
parser.add_argument("--validate_sheet", help="Sheet name to train", type=str, default="46_samples")
parser.add_argument("--pickle_dir", help="Directory to store pickle data", type=str, default="pickle")
parser.add_argument("--tsne", help="Whether overwrite TSNE", action="store_false", default=True)

args = parser.parse_args()

if len(sys.argv) == 1:
    pass

if not os.path.exists(args.file_name) or not os.path.isfile(args.file_name):
    exit("Invalid file: " + args.file)

train_data = pandas.read_excel(args.file_name, sheet_name=args.train_sheet)
validate_data = pandas.read_excel(args.file_name, sheet_name=args.validate_sheet)

train_data = train_data[["관리번호", "Classification", "AL", "PD", "DNA농도(ng/ul)", "Total bacteria", "Aa", "Pg", "Tf", "Td", "Pi", "Fn", "Pa", "Cr", "Ec", "Aa_relative", "Pg_relative", "Tf_relative", "Td_relative", "Pi_relative", "Fn_relative", "Pa_relative", "Cr_relative", "Ec_relative"]]
validate_data = validate_data[["관리번호", "Classification", "AL", "PD", "DNA농도(ng/ul)", "Total bacteria", "Aa", "Pg", "Tf", "Td", "Pi", "Fn", "Pa", "Cr", "Ec", "Aa_relative", "Pg_relative", "Tf_relative", "Td_relative", "Pi_relative", "Fn_relative", "Pa_relative", "Cr_relative", "Ec_relative"]]

train_data.rename(columns={"관리번호": "Number", "DNA농도(ng/ul)": "DNA"}, inplace=True)
validate_data.rename(columns={"관리번호": "Number", "DNA농도(ng/ul)": "DNA"}, inplace=True)

if args.verbose:
    print("Train data:")
    print(train_data)
    print("validate data:")
    print(validate_data)

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

train_tsne_pickle = os.path.join(args.pickle_dir, "train_tsne.pkl")
if os.path.exists(train_tsne_pickle) and args.tsne:
    with open(train_tsne_pickle, "rb") as f:
        train_tsne_data = pickle.load(f)
else:
    tmp_data = train_data[["Total bacteria", "Aa", "Pg", "Tf", "Td", "Pi", "Fn", "Pa", "Cr", "Ec", "Aa_relative", "Pg_relative", "Tf_relative", "Td_relative", "Pi_relative", "Fn_relative", "Pa_relative", "Cr_relative", "Ec_relative"]]
    train_tsne_data = sklearn.manifold.TSNE(n_components=2, random_state=0).fit_transform(tmp_data)

    with open(train_tsne_pickle, "wb") as f:
        pickle.dump(train_tsne_data, f)

validate_tsne_pickle = os.path.join(args.pickle_dir, "validate_tsne.pkl")
if os.path.exists(validate_tsne_pickle) and args.tsne:
    with open(validate_tsne_pickle, "wb") as f:
        validate_tsne_data = pickle.load(f)
else:
    tmp_data = validate_data[["Total bacteria", "Aa", "Pg", "Tf", "Td", "Pi", "Fn", "Pa", "Cr", "Ec", "Aa_relative", "Pg_relative", "Tf_relative", "Td_relative", "Pi_relative", "Fn_relative", "Pa_relative", "Cr_relative", "Ec_relative"]]
    validate_tsne_data = sklearn.manifold.TSNE(n_components=2, random_state=0).fit_transform(tmp_data)

    with open(validate_tsne_pickle, "wb") as f:
        pickle.dump(validate_tsne_data, f)
