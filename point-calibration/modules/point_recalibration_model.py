import torch
from metrics import Metrics
from pytorch_lightning.core.lightning import LightningModule
from sigmoid import (
    SigmoidFlowND,
    SigmoidFlowNDSingleMLP,
    SigmoidFlowNDSingleMLPDropout,
    SigmoidFlowNDMonotonic,
)
from composition import CompositionDist
from losses import PointCalibrationLoss
import torch
import torch.nn as nn
import math


class PointRecalibrationModel(LightningModule):
    def __init__(
        self,
        datasets,
        y_scale,
        n_in=3,
        n_layers=1,
        n_dim=100,
        n_bins=20,
        flow_type=None,
        learning_rate=1e-3,
    ):
        super().__init__()
        self.y_scale = y_scale
        self.n_bins = n_bins
        (
            self.train_dist,
            self.y_train,
            self.val_dist,
            self.y_val,
            self.test_dist,
            self.y_test,
        ) = datasets
        self.n_bins_test = self.y_train.shape[0]
        self.train_loss = PointCalibrationLoss(discretization=n_bins, y=self.y_train)
        if self.val_dist:
            self.val_loss = PointCalibrationLoss(discretization=n_bins, y=self.y_val)
        self.test_loss = PointCalibrationLoss(
            discretization=self.n_bins_test, y=self.y_test
        )

        self.learning_rate = learning_rate
        if flow_type == None:
            self.sigmoid_flow = SigmoidFlowND(
                n_in=n_in, num_layers=n_layers, n_dim=n_dim
            )
        elif flow_type == "single_mlp":
            self.sigmoid_flow = SigmoidFlowNDSingleMLP(
                n_in=n_in, num_layers=n_layers, n_dim=n_dim
            )

    def training_step(self, batch, batch_idx):
        self.train()
        comp = CompositionDist(self.sigmoid_flow, self.train_dist.to(self.device))
        l = self.train_loss(self.y_train.to(self.device), comp)
        tensorboard_logs = {"train_loss": l}
        return {"loss": l, "log": tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def validation_step(self, batch, batch_idx):
        self.eval()
        comp = CompositionDist(self.sigmoid_flow, self.val_dist.to(self.device))
        l = self.val_loss(self.y_val.to(self.device), comp)
        metrics = Metrics(
            comp, self.y_val.to(self.device), self.y_scale, discretization=self.n_bins
        )
        pce = metrics.point_calibration_error_uniform_mass()
        dic = {}
        dic["point_calibration_error_uniform_mass"] = pce
        dic["val_loss"] = l
        return dic

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {}
        for key in outputs[0]:
            cal = torch.stack([x[key] for x in outputs]).mean()
            tensorboard_logs[key] = cal

        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        comp = CompositionDist(self.sigmoid_flow, self.test_dist.to(self.device))
        l = self.test_loss(self.y_test.to(self.device), comp)
        metrics = Metrics(
            comp,
            self.y_test.to(self.device),
            self.y_scale,
        )
        dic = {
            "test_loss": self.test_loss(self.y_test.to(self.device), comp),
        }
        dic2 = metrics.get_metrics(decision_making=True)
        dic.update(dic2)
        return dic

    def backward(self, loss, optimizer, opt_idx):
        # do a custom way of backward
        loss.backward(retain_graph=True)

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        tensorboard_logs = {}
        for key in outputs[0]:
            if key not in ["all_err", "all_loss", "all_y0", "all_c"]:
                cal = torch.stack([x[key] for x in outputs]).mean()
                tensorboard_logs[key] = cal
                setattr(self, key, float(cal))
            else:
                setattr(self, key, outputs[0][key])

        # Record val PCE and decision loss gap
        if self.val_dist:
            comp = CompositionDist(self.sigmoid_flow, self.val_dist.to(self.device))
            l = self.val_loss(self.y_val.to(self.device), comp)
            metrics = Metrics(
                comp,
                self.y_val.to(self.device),
                self.y_scale,
                discretization=self.n_bins,
            )
            dic = metrics.get_metrics(decision_making=True)
            setattr(
                self,
                "val_point_calibration_error",
                dic["point_calibration_error"].item(),
            )
            setattr(self, "val_true_vs_pred_loss", dic["true_vs_pred_loss"].item())

        return {"test_loss": avg_loss, "log": tensorboard_logs}
