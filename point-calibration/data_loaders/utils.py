from torch.utils.data import DataLoader
import torch


def get_dataloaders(train, val, test, batch_size=None):
    r"""
    Transform numpy arrays of train and test data into Pytorch dataloaders for
    train, validation, and test. (Splits the training data to generate a validation
    set).

    Arguments:
        X_train (numpy.ndarray): training features
        y_train (numpy.ndarray): training label
        X_test (numpy.ndarray): test features
        y_test (numpy.ndarray): test labels
        split_seed (int): random seed to generate train/validation split


    Returns:
        train (Pytorch dataloader): training data.
        val (Pytorch dataloader):  validation data.
        test (Pytorch dataloader):  with test data.
        in_size (tuple): feature shape
        target_size (tuple): target shape
    """
    if batch_size == None:
        train_loader = DataLoader(train, batch_size=len(train), shuffle=True)
    else:
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)

    if len(val) != 0:
        val_loader = DataLoader(val, batch_size=len(val), shuffle=False)
    else:
        val_loader = None
    test_loader = DataLoader(test, batch_size=len(test), shuffle=False)
    return train_loader, val_loader, test_loader
