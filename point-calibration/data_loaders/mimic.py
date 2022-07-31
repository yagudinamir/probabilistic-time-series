import pandas as pd
from data_loaders.utils import get_dataloaders
import numpy as np
from torch.utils.data import random_split, DataLoader, TensorDataset
import torch
import os


def _load_mimic_los():
    df = pd.read_excel("data_loaders/data/mimic/mimic_050221.xlsx")
    df2 = pd.read_excel("data_loaders/data/mimic/mimic_050221_2.xlsx")
    total_df = pd.concat([df, df2])
    impute_vals = {
        "cap_refill": 0.0,
        "bp_diastolic": 59.0,
        "fio2": 0.21,
        "gcs_eye": 4,
        "gcs_motor": 6,
        "gcs_total": 15,
        "gcs_verbal": 5,
        "glucose": 128.0,
        "heart_rate": 86.0,
        "height_cm": 170,
        "bp_mean": 77.0,
        "o2sat": 98.0,
        "resp_rate": 19,
        "bp_systolic": 118.0,
        "temp_fahren": 97.88,
        "weight_lbs": 81.0,
        "ph": 7.4,
    }

    total_df = total_df.fillna(impute_vals)

    features = [
        "cap_refill",
        "bp_diastolic",
        "bp_systolic",
        "bp_mean",
        "fio2",
        "gcs_eye",
        "gcs_verbal",
        "gcs_motor",
        "gcs_total",
        "glucose",
        "heart_rate",
        "height_cm",
        "o2sat",
        "resp_rate",
        "temp_fahren",
        "weight_lbs",
        "ph",
    ]
    return total_df[features].to_numpy(), total_df["los"].to_numpy()


def get_mimic_datasets(
    name, split_seed=0, test_fraction=0.10, train_frac=1.0, combine_val_train=False
):
    r"""
    Returns a MIMIC regression dataset in the form of numpy arrays.

    Arguments:
        name (str): name of the dataset
        split_seed (int): seed used to generate train/test split
        test_fraction (float): fraction of the dataset used for the test set


    Returns:
        X_train (numpy.ndarray): training features
        y_train (numpy.ndaray): training label
        X_test (numpy.ndarray): test features
        y_test (numpy.ndarray): test labels
        y_train_scale (float): standard deviation of training labels
    """
    # load full dataset
    load_funs = {"mimic_los": _load_mimic_los}

    print("Loading dataset {}....".format(name))

    X, y = load_funs[name]()
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    # We create the train and test sets with 90% and 10% of the data

    if split_seed == -1:  # Do not shuffle!
        permutation = range(X.shape[0])
    else:
        rs = np.random.RandomState(split_seed)
        permutation = rs.permutation(X.shape[0])

    size_train = int(np.round(X.shape[0] * (1 - test_fraction)))
    index_train = permutation[0:size_train]
    index_test = permutation[size_train:]

    X_train = X[index_train, :]
    X_test = X[index_test, :]
    y_train = y[index_train, None]
    y_test = y[index_test, None]

    if train_frac != 1.0:
        rs = np.random.RandomState(split_seed)
        permutation = rs.permutation(X_train.shape[0])
        n_train = int(train_frac * len(X_train))
        X_train = X_train[:n_train]
        y_train = y_train[:n_train]

    if split_seed == -1:  # Do not shuffle!
        permutation = range(X_train.shape[0])
    else:
        rs = np.random.RandomState(split_seed)
        permutation = rs.permutation(X_train.shape[0])

    if combine_val_train:
        val_fraction = 0.0
    else:
        val_fraction = 0.10
    size_train = int(np.round(X_train.shape[0] * (1 - val_fraction)))
    index_train = permutation[0:size_train]
    index_val = permutation[size_train:]

    X_new_train = X_train[index_train, :]
    X_val = X_train[index_val, :]

    y_new_train = y_train[index_train]
    y_val = y_train[index_val]

    print("Done loading dataset {}".format(name))

    def standardize(data):
        mu = data.mean(axis=0, keepdims=1)
        scale = data.std(axis=0, keepdims=1)
        scale[scale < 1e-10] = 1.0

        data = (data - mu) / scale
        return data, mu, scale

    # Standardize
    X_new_train, x_train_mu, x_train_scale = standardize(X_new_train)
    X_test = (X_test - x_train_mu) / x_train_scale
    y_new_train, y_train_mu, y_train_scale = standardize(y_new_train)
    y_test = (y_test - y_train_mu) / y_train_scale
    X_val = (X_val - x_train_mu) / x_train_scale
    y_val = (y_val - y_train_mu) / y_train_scale

    train = TensorDataset(
        torch.Tensor(X_new_train).type(torch.float64),
        torch.Tensor(y_new_train).type(torch.float64),
    )

    val = TensorDataset(
        torch.Tensor(X_val).type(torch.float64),
        torch.Tensor(y_val).type(torch.float64),
    )

    test = TensorDataset(
        torch.Tensor(X_test).type(torch.float64),
        torch.Tensor(y_test).type(torch.float64),
    )
    in_size = X_train[0].shape
    target_size = y_train[0].shape

    return train, val, test, in_size, target_size, y_train_scale


def get_mimic_dataloaders(
    name,
    split_seed=0,
    test_fraction=0.1,
    batch_size=None,
    train_frac=1.0,
    combine_val_train=False,
):
    r"""
    Returns a MIMIC regression dataset in the form of Pytorch dataloaders
    for train, validation, and test. Also returns the sizes of features and label
    and the standard deviation of the training labels.

    Arguments:
        name (str): name of the dataset
        split_seed (int): seed used to generate train/test split
        test_fraction (float): fraction of the dataset used for the test set

    Returns:
        train_loader (Pytorch dataloader): training data.
        val_loader (Pytorch dataloader): validation data.
        test_loader (Pytorch dataloader): test data.
        in_size (tuple): feature shape
        target_size (tuple): target shape
        y_train_scale (float): standard deviation of training labels.

    """
    train, val, test, in_size, target_size, y_train_scale = get_mimic_datasets(
        name,
        split_seed=split_seed,
        test_fraction=test_fraction,
        train_frac=train_frac,
        combine_val_train=combine_val_train,
    )

    train_loader, val_loader, test_loader = get_dataloaders(
        train, val, test, batch_size
    )
    return train_loader, val_loader, test_loader, in_size, target_size, y_train_scale

if __name__ == "__main__":
    print(_load_mimic_los()[0].shape)

