from re import A
from data_loaders.utils import get_dataloaders

"""
IO module for UCI datasets for regression

Source: https://github.com/aamini/evidential-regression/blob/c0823f18ff015f5eb46a23f0039f4d62b76bc8d1/data_loader.py
"""
import numpy as np
import pandas as pd
import os
import h5py
import torch
from torch.utils.data import random_split, DataLoader, TensorDataset
from sklearn.datasets import load_boston
from data_loaders.utils import get_dataloaders


def get_nasdaq_datasets(name, split_seed=0, test_fraction=0.10, train_frac=1.0, combine_val_train=False):
    # for nasdaq dataset we are testing only iterative_point, average & no calibration for which we don't need x labels
    mean_train = np.load("../notebooks/prediction-stats/mean_train.npy").astype(np.float32)
    var_train = np.load("../notebooks/prediction-stats/var_train.npy").astype(np.float32)
    y_train = np.load("../notebooks/prediction-stats/targets_train.npy").astype(np.float32)
    mean_val = np.load("../notebooks/prediction-stats/mean_val.npy").astype(np.float32)
    var_val = np.load("../notebooks/prediction-stats/var_val.npy").astype(np.float32)
    y_val = np.load("../notebooks/prediction-stats/targets_val.npy").astype(np.float32)
    mean_test = np.load("../notebooks/prediction-stats/mean_test.npy").astype(np.float32)
    var_test = np.load("../notebooks/prediction-stats/var_test.npy").astype(np.float32)
    y_test = np.load("../notebooks/prediction-stats/targets_test.npy").astype(np.float32)

    for tensor in [mean_train, var_train, y_train, mean_val, var_val, y_val, mean_test, var_test, y_test]:
        tensor.shape = (-1, 1)
    print("y_train.shape, y_val.shape, y_test.shape", y_train.shape, y_val.shape, y_test.shape)

    y_scale = np.array([[1.0]]).astype(np.float64)
    print("y_test.dtype", y_test.dtype)
    print("y_scale.dtype", y_scale.dtype)

    # if split_seed == -1:  # Do not shuffle!
    #     permutation = range(y_train.shape[0])
    # else:
    #     rs = np.random.RandomState(split_seed)
    #     permutation = rs.permutation(y_train.shape[0])
    # mean_train = mean_train[permutation]
    # var_train = var_train[permutation]
    # y_train = y_train[permutation]

    train = TensorDataset(
        torch.Tensor(np.zeros_like(y_train)).type(torch.float64),
        torch.Tensor(y_train).type(torch.float64),
        torch.Tensor(mean_train).type(torch.float64),
        torch.Tensor(var_train).type(torch.float64),
    )

    val = TensorDataset(
        torch.Tensor(np.zeros_like(y_val)).type(torch.float64),
        torch.Tensor(y_val).type(torch.float64),
        torch.Tensor(mean_val).type(torch.float64),
        torch.Tensor(var_val).type(torch.float64),
    )

    test = TensorDataset(
        torch.Tensor(np.zeros_like(y_test)).type(torch.float64),
        torch.Tensor(y_test).type(torch.float64),
        torch.Tensor(mean_test).type(torch.float64),
        torch.Tensor(var_test).type(torch.float64),
    )
    in_size = None
    target_size = y_train[0].shape

    return train, val, test, in_size, target_size, y_scale


def get_nasdaq_dataloaders(
    name,
    split_seed=0,
    test_fraction=0.1,
    batch_size=None,
    train_frac=1.0,
    combine_val_train=False,
):
    train, val, test, in_size, target_size, y_train_scale = get_nasdaq_datasets(
        name,
        split_seed=split_seed,
        test_fraction=test_fraction,
        train_frac=train_frac,
        combine_val_train=combine_val_train,
    )
    assert batch_size is None  # temporary solution for nasdaq dataset
    train_loader, val_loader, test_loader = get_dataloaders(train, val, test, batch_size)
    return train_loader, val_loader, test_loader, in_size, target_size, y_train_scale


def get_nasdaq_model_predictions(model, dist_class, train, val, test, cuda=False):
    assert model == "nasdaq"

    def dataset_dist(data):
        assert len(data) == 1
        for batch in data:
            x, y, mu, var = batch
            params = (mu, var)
            if cuda:
                x = x.to(device)
                y = y.to(device)
                params = [param.flatten().detach() for param in params]
                y = y.flatten()
            else:
                params = [param.detach().cpu().flatten() for param in params]
                y = y.flatten().detach().cpu()

            dist = dist_class(tuple(params))
            return dist, y

    train_dist, y_train = dataset_dist(train)
    print("got train dist")
    if val:
        val_dist, y_val = dataset_dist(val)
    else:
        val_dist = None
        y_val = None
    test_dist, y_test = dataset_dist(test)

    return train_dist, y_train, val_dist, y_val, test_dist, y_test
