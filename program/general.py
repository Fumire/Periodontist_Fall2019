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


if __name__ == "__main__":
    pass
