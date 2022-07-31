from pytorch_lightning.core.lightning import LightningModule
import torch
import numpy as np


class ClassifierModel(LightningModule):
    def __init__(self, input_size):
        super().__init__()
        torch.set_default_tensor_type(torch.DoubleTensor)
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, 50), torch.nn.ReLU(), torch.nn.Linear(50, 2)
        )

        self.test_loss = 0
        self.rmse = 0
        self.log_likelihood = 0
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        probs = self.layers(x)
        return probs

    def training_step(self, batch, batch_idx):
        x, y = batch
        probs = self(x)
        loss = self.loss(probs, y)
        tensorboard_logs = {
            "train_loss": loss,
        }

        return {"loss": loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def validation_step(self, batch, batch_idx):

        x, y = batch
        probs = self(x)

        return {"val_loss": self.loss(probs, y)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}

        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        probs = self(x)

        dic = {
            "test_loss": self.loss(probs, y),
        }
        softmax = torch.nn.Softmax(dim=1)
        default_prob = softmax(probs)[:, 1]
        data = torch.hstack((x, default_prob.reshape(-1, 1)))
        with open("data_loaders/data/GiveMeSomeCredit/test.csv", "a+") as f:
            np.savetxt(f, data.detach().cpu().numpy(), delimiter=",")

        return dic

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()

        tensorboard_logs = {}
        for key in outputs[0]:
            cal = torch.stack([x[key] for x in outputs]).mean()
            tensorboard_logs[key] = cal
            setattr(self, key, float(cal))
        softmax = torch.nn.Softmax(dim=1)

        for batch in self.train_dataloader():
            x, y = batch
            probs = self(x.to(self._device))
            default_prob = softmax(probs)[:, 1].detach().cpu()
            data = torch.hstack((x.detach().cpu(), default_prob.reshape(-1, 1)))
            with open("data_loaders/data/GiveMeSomeCredit/train.csv", "a+") as f:
                np.savetxt(f, data.numpy(), delimiter=",")

        for batch in self.val_dataloader():
            x, y = batch
            probs = self(x.to(self._device))
            default_prob = softmax(probs)[:, 1].detach().cpu()
            data = torch.hstack((x.detach().cpu(), default_prob.reshape(-1, 1)))
            with open("data_loaders/data/GiveMeSomeCredit/val.csv", "a+") as f:
                np.savetxt(f, data.detach().cpu().numpy(), delimiter=",")

        return {"test_loss": avg_loss, "log": tensorboard_logs}
