# METRICS
from typing import Union
import pandas as pd
import math

def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size 
    true = (y_hat == y).sum() ## count of correct predictions
    tot = y.size
    if tot == 0: ## to avoid zero division
        return 1
    acc = true / tot
    return acc
    pass

# # CrossCheck:
# y_hat = pd.Series([1, 0, 1, 1, 0])
# y = pd.Series([1, 1, 1, 0, 0])
# acc = accuracy(y_hat, y)
# print(acc)

def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    assert y_hat.size == y.size
    assert cls in y.unique() # To check if the specified class is present in the unique values of y
    true = ((y_hat == cls) & (y == cls)).sum()
    true_y_hat = (y_hat == cls).sum()
    if true_y_hat == 0:
        return 1
    prec = true / true_y_hat
    return prec
    pass
    
# # # CrossCheck:
# y_hat = pd.Series([1, 0, 1, 1, 0])
# y = pd.Series([1, 1, 1, 0, 0])
# cls = 1
# prec = precision(y_hat, y, cls)
# print(prec)

def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.size == y.size
    assert cls in y.unique()
    true = ((y_hat == cls) & (y == cls)).sum()
    true_y = (y == cls).sum()
    if true_y == 0:
        return 1
    recall = true / true_y
    return recall
    pass
    
# # CrossCheck:
# y_hat = pd.Series([1, 0, 1, 1, 0])
# y = pd.Series([1, 1, 1, 1, 0])
# cls = 1
# rec = recall(y_hat, y, cls)
# print(rec)

def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    assert y_hat.size == y.size
    err = ((y_hat - y)**2).sum()
    rms_val = (err/y.size)**0.5
    return rms_val
    pass

# # CrossCheck:
# y_hat = pd.Series([1, 0, 1, 1, 0])
# y = pd.Series([1, 1, 1, 1, 0])
# ans = rmse(y_hat, y)
# print(ans)

def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.size == y.size
    err = (abs(y_hat - y)).sum()
    mae = err / y.size
    return mae
    pass

# # CrossCheck:
# y_hat = pd.Series([1, 0, 1, 1, 0])
# y = pd.Series([1, 1, 1, 1, 0])
# ans = mae(y_hat, y)
# print(ans)
