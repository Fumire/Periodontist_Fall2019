import argparse
import itertools
import os

absolute_values = sorted(["Aa", "Pg", "Tf", "Td", "Pi", "Fn", "Pa", "Cr", "Ec"])
relative_values = sorted(["Aa_relative", "Pg_relative", "Tf_relative", "Td_relative", "Pi_relative", "Fn_relative", "Pa_relative", "Cr_relative", "Ec_relative"])
whole_values = sorted(absolute_values + relative_values)

classes = ["Healthy", "Slight", "Moderate", "Severe", "Acute"]
two_class_combinations = itertools.combinations(classes, 2)
three_class_combinations = itertools.combinations(classes, 3)

default_result_directory = os.path.realpath("../results")


def check_exist(file_name):
    directory = os.path.dirname(file_name)
    if os.path.isfile(directory):
        raise ValueError
    elif os.path.isdir(directory):
        pass
    else:
        os.makedirs(directory)
    return file_name


def class_to_num(given_classes):
    for something in given_classes:
        if something not in classes:
            raise KeyError(something)
    return sum(list(map(lambda x: 2 ** x[0], list(filter(lambda x: x[1] in given_classes, list(enumerate(classes)))))))


def num_to_class(num):
    if num > 2 ** len(classes) or num < 0:
        raise ValueError
    return list(map(lambda x: x[1], list(filter(lambda x: 2 ** x[0] & num, list(enumerate(classes))))))


def bacteria_to_num(bacteria):
    for something in bacteria:
        if something not in whole_values:
            raise KeyError(something)
    return sum(list(map(lambda x: 2 ** x[0], list(filter(lambda x: x[1] in bacteria, list(enumerate(whole_values)))))))


def num_to_bacteria(num):
    if num > 2 ** len(whole_values) or num < 0:
        raise ValueError
    return list(map(lambda x: x[1], list(filter(lambda x: 2 ** x[0] & num, list(enumerate(whole_values))))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--classes", help="class to choose", nargs="+", choices=classes, default=[])
    parser.add_argument("-b", "--bacteria", help="Bacteria to choose", nargs="+", choices=whole_values, default=[])

    parser.add_argument("--classnum", help="Number changed into class", type=int, default=0)
    parser.add_argument("--bacterianum", help="Number changed into bacteria", type=int, default=0)

    args = parser.parse_args()

    print("class to num:", class_to_num(args.classes))
    print("bacteria to num:", bacteria_to_num(args.bacteria))
    print("num to class", num_to_class(args.classnum))
    print("num to bacteria", num_to_bacteria(args.bacterianum))
