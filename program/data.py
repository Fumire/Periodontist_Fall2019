import argparse
import os
import pandas
import general

default_xlsx_file = os.path.realpath("../data/Periodontitis_input_dataset_from_784samples_and_additional_54samples_20190730.xlsx")
default_sheet_name = ["730_samples", "54_samples"]
default_csv_directory = os.path.join(general.default_result_directory, "csv")

extra_columns = ["관리번호", "Classification", "AL", "PD", "DNA농도(ng/ul)", "Total bacteria"]
rename_columns = {"관리번호": "ID", "DNA농도(ng/ul)": "DNA"}


def get_data(input_file=default_xlsx_file, sheet_name=default_sheet_name, read_columns=general.whole_values, output_file=None):
    if not read_columns:
        raise ValueError("Empty column")
    elif not sheet_name:
        raise ValueError("Empty sheet name")

    if output_file is None:
        output_file = os.path.join(default_csv_directory, "+".join(read_columns) + ".csv")

    data = pandas.concat(pandas.read_excel(input_file, sheet_name=sheet_name), ignore_index=True)
    data = data[extra_columns + read_columns]
    data.rename(columns=rename_columns, inplace=True)
    data["Classification"] = list(map(lambda x: {"Healthy": "Healthy", "CP_E": "Slight", "CP_M": "Moderate", "CP_S": "Severe", "AP": "Acute"}[x], data["Classification"]))

    data.to_csv(general.check_exist(output_file), index=False)
    return data


def remove_ap(input_file=None, output_file=None):
    if input_file is None:
        raise ValueError(input_file)
    elif not os.path.isfile(input_file):
        raise ValueError(input_file)

    data = pandas.read_csv(input_file)
    data = data.loc[~(data["Classification"] == "Acute")]

    data.to_csv(general.check_exist(input_file.replace(".csv", ".remove_ap.csv")), index=False)
    return data


def select_data(input_file=None, output_file=None, classes=None, bacteria=None):
    if input_file is None:
        raise ValueError(input_file)
    elif not os.path.isfile(input_file):
        raise ValueError(input_file)

    data = pandas.read_csv(input_file)

    if classes is None:
        classes = sorted(set(data["Classification"]))
    else:
        for something in classes:
            if something not in set(data["Classification"]):
                raise KeyError(something)
        classes.sort()

    if bacteria is None:
        bacteria = list(filter(lambda x: x in list(data.columns), general.whole_values))
    else:
        for something in bacteria:
            if something not in data.columns:
                raise KeyError(something)
        bacteria.sort()

    if output_file is None:
        output_file = input_file.replace(".csv", "") + "." + str(general.class_to_num(classes)) + "-" + str(general.bacteria_to_num(bacteria)) + ".csv"

    data = data.loc[(data["Classification"].isin(classes))]
    data = data[["ID"] + bacteria]

    data.to_csv(output_file, index=False)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    group1 = parser.add_mutually_exclusive_group(required=True)
    group1.add_argument("--xlsx", help="Read file is XLSX format", action="store_true", default=False)
    group1.add_argument("--remove_ap", help="Remove AP in CSV", action="store_true", default=False)
    group1.add_argument("--select", help="Select amongst CSV", action="store_true", default=False)
    group1.add_argument("--number", help="Input is number only", action="store_true", default=False)

    parser.add_argument("-v", "--verbose", help="Verbose output", action="store_true", default=False)
    parser.add_argument("-i", "--input_file", help="File name to input", default=None)
    parser.add_argument("-o", "--output_file", help="File name to output", default=None)
    parser.add_argument("-b", "--bacteria", help="Bacteria to analyze", choices=general.whole_values, nargs="*", default=general.whole_values)
    parser.add_argument("-s", "--sheet", help="Sheet name to read", choices=default_sheet_name, nargs="*", default=default_sheet_name)
    parser.add_argument("-c", "--classes", help="Class name to select", choices=general.classes, nargs="*", default=general.classes)

    args = parser.parse_args()

    if args.xlsx:
        data = get_data(read_columns=args.bacteria, output_file=args.output_file, sheet_name=args.sheet, input_file=default_xlsx_file)
    elif args.remove_ap:
        data = remove_ap(input_file=args.input_file, output_file=args.output_file)
    elif args.select:
        data = select_data(input_file=args.input_file, output_file=None, classes=args.classes, bacteria=args.bacteria)
    elif args.number:
        tmp = args.output_file.split(".")
        numbers = list(map(int, tmp[-2].split("-")))
        data = select_data(input_file=args.input_file, output_file=args.output_file, classes=general.num_to_class(numbers[0]), bacteria=general.num_to_bacteria(numbers[1]))
    else:
        exit("Something went wrong")

    if args.verbose:
        print(data)
