import argparse
import multiprocessing
import sklearn.ensemble
import sklearn.gaussian_process
import sklearn.neighbors
import sklearn.neural_network
import sklearn.svm
import sklearn.tree
import general

classifiers = [("KNeighbors", sklearn.neighbors.KNeighborsClassifier(algorithm="brute", n_jobs=1)), ("SVC", sklearn.svm.SVC(probability=True, decision_function_shape="ovr", random_state=0)), ("Gaussian", sklearn.gaussian_process.GaussianProcessClassifier(max_iter_predict=2**30, random_state=0, multi_class="one_vs_rest", n_jobs=1)), ("DecisionTree", sklearn.tree.DecisionTreeClassifier(random_state=0)), ("RandomeForest", sklearn.ensemble.RandomForestClassifier(random_state=0, n_jobs=1, class_wight="balanced")), ("NeuralNetwork", sklearn.neural_network.MLPClassifier(max_iter=2 ** 30, random_state=0, early_stopping=True)), ("AdaBoost", sklearn.ensemble.AdaBoostClassifier(random_state=0))]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-v", "--verbose", help="Verbose output", action="store_true", default=False)
    parser.add_argument("-i", "--input_file", help="File name to input", default=None)
    parser.add_argument("-o", "--output_file", help="File name to output", default=None)
    parser.add_argument("-j", "--jobs", help="Number of parallel jobs", type=int, default=30)

    args = parser.parse_args()
