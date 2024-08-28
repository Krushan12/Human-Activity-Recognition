import pandas as pd
import numpy as np
from tree.base import DecisionTree  # Assuming you have the DecisionTree class defined in tree.base
from metrics import *
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)

# Dividing the sample into training and test data sets of 70% and 30%
m = int(X.shape[0] * 0.7)
X_train = X[:m]
y_train = y[:m]
X_test = X[m:]
y_test = y[m:]

columns = [str(i) for i in range(len(X[0]))]

X_train = pd.DataFrame(X_train, columns=columns)
y_train = pd.Series(y_train, dtype="category")

tree = DecisionTree(criterion="information_gain", max_depth=5)
tree.fit(X_train, y_train)
X_test = pd.DataFrame(X_test, columns=columns)
y_test = pd.Series(y_test, dtype="int64")
y_hat = tree.predict(X_test,max_depth=5)
print("Answer A")
print("Accuracy:", accuracy(y_test, y_hat))

for i in y_test.unique():
    # Per-class precision.
    print("Precision for class {} is: {}".format(i, precision(y_hat.copy(), y_test.copy(), i)))

for i in y_test.unique():
    # Per-class recall.
    print("Recall for class {} is: {}".format(i, recall(y_hat.copy(), y_test.copy(), i)))

print()
print("Answer B")

def cons_df(X_train, y_train, X_valid, y_valid):
    columns = [str(i) for i in range(len(X[0]))]
    X_train = pd.DataFrame(X_train, columns=columns)
    y_train = pd.Series(y_train, dtype="category")
    X_valid = pd.DataFrame(X_valid, columns=columns)
    y_valid = pd.Series(y_valid, dtype="int64")
    return X_train, y_train, X_valid, y_valid


def acc_fold(X_train, y_train, X_test, y_test, depth):
    X_train, y_train, X_test, y_test = cons_df(X_train, y_train, X_test, y_test)
    tree = DecisionTree(criterion="information_gain", max_depth=depth)  # Split based on Inf. Gain
    tree.fit(X_train, y_train)
    y_hat = tree.predict(X_test,max_depth=5)
    return accuracy(y_hat, y_test)


def find_depth_optimal(X, y):
    m = int(0.25 * len(X))
    acc_depth = {}
    n = len(X)

    for depth in range(1, 4):
        acc_depth[depth] = 0

        for i in range(4):
            start_idx = i * m
            end_idx = (i + 1) * m

            if end_idx < n:  # Ensure there are enough data points for this fold
                X_train, y_train = np.concatenate((X[0:start_idx], X[end_idx:n])), np.concatenate((y[0:start_idx], y[end_idx:n]))
                X_valid, y_valid = X[start_idx:end_idx], y[start_idx:end_idx]

                X_train, y_train, X_valid, y_valid = cons_df(X_train, y_train, X_valid, y_valid)

                tree = DecisionTree(criterion="information_gain", max_depth=depth)
                tree.fit(X_train, y_train)

                y_hat = tree.predict(X_valid,max_depth=5)
                acc_depth[depth] += accuracy(y_hat, y_valid)

        # After finding average validation accuracy for each depth, we choose the optimal depth for that fold as the max(avg_validation accuracy).
        acc_depth[depth] = acc_depth[depth] / 4

    depth_opt = max(acc_depth, key=lambda x: acc_depth[x])
    return depth_opt, acc_depth[depth_opt]



def K_fold(X, y):
    K = 5
    n = int(X.shape[0] / K)
    l = len(X)
    ans_acc = {}
    for i in range(K - 1, -1, -1):
        # print("-" * 6 + f"Optimal depth for fold {i+1} " + "-" * 6)
        X_train, y_train = (
            np.concatenate((X[0:i * n], X[(i + 1) * n : l])),
            np.concatenate((y[0:i * n], y[(i + 1) * n : l])),
        )
        X_test, y_test = X[i * n : (i + 1) * n], y[i * n : (i + 1) * n]
        opt_depth, acc_depth = find_depth_optimal(X_train, y_train)
        fold_acc = acc_fold(X_train, y_train, X_test, y_test, opt_depth)
        print(f"For fold: {i}, Optimal depth: {opt_depth}, Average validation accuracy: {acc_depth}, Test accuracy: {fold_acc}")
        ans_acc[i] = [opt_depth, acc_depth, fold_acc]
    return ans_acc


X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5
)

ans_ac = K_fold(X, y)
f = max(ans_ac, key=lambda x: ans_ac[x][1])
print()
print("Optimal depth: ", ans_ac[f][0])
print("Validation accuracy of optimal depth: ", ans_ac[f][1])
print("Test accuracy for optimal depth: ", ans_ac[f][2])

plt.show()
