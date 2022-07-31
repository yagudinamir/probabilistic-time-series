from torch.utils.data import Dataset, DataLoader
import torch


def get_recalibration_dataloaders(
    train_dist, y_train, val_dist, y_val, test_dist, y_test
):
    train_dataset = RecalibrationDataset(train_dist, y_train)
    if val_dist:
        val_dataset = RecalibrationDataset(val_dist, y_val)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    else:
        val_loader = None
    test_dataset = RecalibrationDataset(test_dist, y_test)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_loader, val_loader, test_loader


class RecalibrationDataset(Dataset):
    def __init__(self, dist, labels):
        self.dist_params = dist
        self.labels = labels

    def __getitem__(self, idx):
        return self.labels

    def __len__(self):
        return 1
