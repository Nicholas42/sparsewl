import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from auxiliarymethods.kernel_evaluation import kernel_svm_evaluation, linear_svm_evaluation
from auxiliarymethods.auxiliary_methods import read_lib_svm, normalize_gram_matrix, normalize_feature_vector
import os.path
from os import path as pth
from datatime import datetime


def read_classes(ds_name):
    with open("../datasets/" + ds_name + "/" + ds_name + "_graph_labels.txt", "r") as f:
        classes = [int(i) for i in list(f)]
    f.closed

    return np.array(classes)


def main(dataset):

    path = "./GM/EXP/"
    algorithms = ["LWLC2", "WL1", "GR", "SP", "WLOA", "LWL2",
        "LWLP2", "WL2", "DWL2", "LWL3", "LWLP3", "WL3", "DWL3"]

    for a in algorithms:
        gram_matrices = []
        for i in range(0, 10):
            if not pth.exists(path + dataset + "__" + a + "_" + str(i) + ".gram"):
                continue
            else:
                gram_matrix, _ = read_lib_svm(
                    path + dataset + "__" + a + "_" + str(i) + ".gram")
                gram_matrix = normalize_gram_matrix(gram_matrix)
                classes = read_classes(dataset)
                gram_matrices.append(gram_matrix)

        if gram_matrices != []:
            start = datetime.now()
            acc, acc_train, s_1 = kernel_svm_evaluation(
                gram_matrices, classes, num_repetitions=10)
            end = datetime.now()
            print(f'{a=}', f'{dataset=}', f'{acc=}',
                  f'{acc_train=}', f'{s_1=}')
            print(f'Took {end - start}')
        else:
            print(f"Nothing to do for algorithm {a} on {dataset}.")


if __name__ == "__main__":
    main(sys.argv[1])
