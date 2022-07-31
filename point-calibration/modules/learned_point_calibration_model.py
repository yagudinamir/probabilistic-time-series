import torch
from pytorch_lightning.core.lightning import LightningModule
from metrics import Metrics
from losses import PointCalibrationLoss
from distributions import GaussianDistribution, GaussianLaplaceMixtureDistribution


class LearnedPointCalibrationModel(LightningModule):
    def __init__(self, input_size, y_scale):
        super().__init__()
        torch.set_default_tensor_type(torch.DoubleTensor)
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 5),
        )
        self.loss = PointCalibrationLoss(discretization=100)
        self.y_scale = y_scale

    def forward(self, x):
        x = self.layers(x)
        mu, logvar, loc, logscale, weight = torch.chunk(x, chunks=5, dim=1)
        var = torch.exp(logvar)
        scale = torch.exp(logscale)
        weight = torch.sigmoid(weight)
        return mu, var, loc, scale, weight

    #        mu, logvar = torch.chunk(x, chunks=2, dim=1)
    #        var = torch.exp(logvar)
    #        return mu, var

    def training_step(self, batch, batch_idx):
        x, y = batch
        params = self(x)
        params = [param.flatten() for param in params]
        #        dist = GaussianDistribution(params)
        dist = GaussianLaplaceMixtureDistribution(params)
        l = self.loss(y, dist)
        tensorboard_logs = {"train_loss": l}
        return {"loss": l, "log": tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def validation_step(self, batch, batch_idx):
        x, y = batch
        params = self(x)
        params = [param.flatten() for param in params]
        #        dist = GaussianDistribution(params)
        dist = GaussianLaplaceMixtureDistribution(params)
        loss = self.loss(y, dist)
        cpu_params = tuple(
            [params[i].detach().cpu().flatten() for i in range(len(params))]
        )
        #        dist = GaussianDistribution(cpu_params)
        dist = GaussianLaplaceMixtureDistribution(cpu_params)

        metrics = Metrics(dist, y.detach().cpu(), self.y_scale)
        dic = {}
        dic["val_loss"] = loss
        dic["point_calibration_error"] = metrics.point_calibration_error()
        return dic

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {}
        for key in outputs[0]:
            cal = torch.stack([x[key] for x in outputs]).mean()
            tensorboard_logs[key] = cal
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        params = self(x)
        params = [param.flatten() for param in params]
        #        dist = GaussianDistribution(params)
        dist = GaussianLaplaceMixtureDistribution(params)
        loss = self.loss(y, dist)

        dic = {
            "test_loss": self.loss(y, dist),
        }
        cpu_params = tuple(
            [params[i].detach().cpu().flatten() for i in range(len(params))]
        )
        #        dist = GaussianDistribution(cpu_params)
        dist = GaussianLaplaceMixtureDistribution(cpu_params)
        metrics = Metrics(dist, y.detach().cpu(), self.y_scale)
        dic2 = metrics.get_metrics(decision_making=True)
        dic.update(dic2)
        return dic

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
        return {"test_loss": avg_loss, "log": tensorboard_logs}
