import pandas as pd
from data_loaders.utils import get_dataloaders
import numpy as np
import os
from torch.utils.data import random_split, DataLoader, TensorDataset
import torch

#####################################
# individual data files             #
#####################################
vb_dir = os.path.dirname(__file__)
data_dir = os.path.join(vb_dir, "data/GiveMeSomeCredit")


def process_data(df):
    percent = df.isnull().sum() / df.isnull().count() * 100
    m_per = percent[percent > 10]
    df = df.drop(columns=m_per.index, axis=1)
    for i in df.columns:
        df[i].fillna(df[i].mean(), inplace=True)
    return df


def _load_credit():
    data_file = os.path.join(data_dir, "cs-training.csv")
    train_data = pd.read_csv(data_file, sep=",")
    train_data = process_data(train_data)
    X = train_data.values[:, 2:]
    y = train_data.values[:, 1]

    return X, y


def _load_credit_regression(seed):
    data_file = os.path.join(data_dir, "train.csv")
    train_data = pd.read_csv(data_file, sep=",")
    X_train = train_data.values[:, :-1]
    y_train = train_data.values[:, -1]

    rng = np.random.RandomState(seed)
    noise = rng.normal(0, 0.05, size=len(y_train))
    y_train = y_train + noise

    test_data_file = os.path.join(data_dir, "test.csv")
    test_data = pd.read_csv(test_data_file, sep=",")
    X_test = test_data.values[:, :-1]
    y_test = test_data.values[:, -1]
    noise = rng.normal(0, 0.05, size=len(y_test))
    y_test = y_test + noise

    val_data_file = os.path.join(data_dir, "val.csv")
    val_data = pd.read_csv(val_data_file, sep=",")
    X_val = val_data.values[:, :-1]
    y_val = val_data.values[:, -1]
    noise = rng.normal(0, 0.05, size=len(y_val))
    y_val = y_val + noise

    return X_train, y_train, X_test, y_test, X_val, y_val


def get_credit_regression_dataloader(split_seed, batch_size):
    X_train, y_train, X_test, y_test, X_val, y_val = _load_credit_regression(split_seed)

    def standardize(data):
        mu = data.mean(axis=0, keepdims=1)
        scale = data.std(axis=0, keepdims=1)
        scale[scale < 1e-10] = 1.0

        data = (data - mu) / scale
        return data, mu, scale

    y_train, y_train_mu, y_train_scale = standardize(y_train)
    y_test = (y_test - y_train_mu) / y_train_scale
    y_val = (y_val - y_train_mu) / y_train_scale
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)

    train = TensorDataset(
        torch.Tensor(X_train).type(torch.float64),
        torch.Tensor(y_train).type(torch.float64),
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

    train_loader, val_loader, test_loader = get_dataloaders(
        train, val, test, batch_size
    )
    return train_loader, val_loader, test_loader, in_size, target_size, y_train_scale


def get_credit_dataloader(split_seed, batch_size):
    X, y = _load_credit()

    test_fraction = 0.3
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
    y_train = y[index_train]
    y_test = y[index_test]

    if split_seed == -1:  # Do not shuffle!
        permutation = range(X_train.shape[0])
    else:
        rs = np.random.RandomState(split_seed)
        permutation = rs.permutation(X_train.shape[0])
    val_fraction = 0.25
    size_train = int(np.round(X_train.shape[0] * (1 - val_fraction)))
    index_train = permutation[0:size_train]
    index_val = permutation[size_train:]

    X_new_train = X_train[index_train, :]
    X_val = X_train[index_val, :]

    y_new_train = y_train[index_train]
    y_val = y_train[index_val]

    def standardize(data):
        mu = data.mean(axis=0, keepdims=1)
        scale = data.std(axis=0, keepdims=1)
        scale[scale < 1e-10] = 1.0

        data = (data - mu) / scale
        return data, mu, scale

    # Standardize
    X_new_train, x_train_mu, x_train_scale = standardize(X_new_train)
    X_test = (X_test - x_train_mu) / x_train_scale
    X_val = (X_val - x_train_mu) / x_train_scale

    train = TensorDataset(
        torch.Tensor(X_new_train).type(torch.float64),
        torch.Tensor(y_new_train).type(torch.long),
    )

    val = TensorDataset(
        torch.Tensor(X_val).type(torch.float64),
        torch.Tensor(y_val).type(torch.long),
    )

    test = TensorDataset(
        torch.Tensor(X_test).type(torch.float64),
        torch.Tensor(y_test).type(torch.long),
    )
    in_size = X_train[0].shape
    target_size = y_train[0].shape

    train_loader, val_loader, test_loader = get_dataloaders(
        train, val, test, batch_size
    )
    return train_loader, val_loader, test_loader, in_size, target_size
