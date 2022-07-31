import torch
from data_loaders import get_simulated_dataloaders, get_uci_dataloaders
import torch.distributions as D
import numpy as np
import torch
import os
import csv
from tqdm import tqdm


class DecisionMaker:
    def __init__(self, c, y0, dist):
        self.c_00 = 0
        self.c_11 = 0
        x = 1 / c - 1
        y = 1
        self.c_01 = (x / (x + y)) * 10
        self.c_10 = (y / (x + y)) * 10
        self.y0 = y0
        self.c = c
        if "Composition" in dist.__class__.__name__:
            self.cdf_vals = (
                dist.cdf(torch.Tensor([self.y0]).to(dist.f.dist_mean.get_device()))
                .detach()
                .cpu()
            )
        else:
            self.cdf_vals = dist.cdf(torch.Tensor([self.y0]))

    def predicted_loss(self, alpha):
        term1 = torch.mean((self.cdf_vals <= alpha) * (self.cdf_vals)) * self.c_01
        term2 = torch.mean((self.cdf_vals > alpha) * (1 - self.cdf_vals)) * self.c_10

        return term1 + term2

    def true_loss(self, y, alpha):
        term1 = (
            torch.mean(((y >= self.y0) & (self.cdf_vals >= alpha)).float())
        ) * self.c_10
        term2 = (
            torch.mean(((y < self.y0) & (self.cdf_vals < alpha)).float())
        ) * self.c_01

        return term1 + term2

    def predict_min_loss(self):
        bayes_opt = self.c
        loss = self.predicted_loss(bayes_opt)
        return loss, bayes_opt

    def compute_decision_loss(self, y):
        true_loss_pred_alpha = self.true_loss(y.detach().cpu(), self.c)
        return true_loss_pred_alpha

    def compute_decision_gap(self, y):
        gap = 0
        actions = torch.linspace(0.05, 0.95, 50)
        for i in range(len(actions)):
            pred_loss = self.predicted_loss(actions[i])
            true_loss = self.true_loss(y.detach().cpu(), actions[i])
            gap += torch.abs(pred_loss - true_loss)
        return gap / len(actions)


def simulate_decision_making(decision_makers, dist, y):
    total_err = 0.0
    total_loss = 0.0
    all_loss = []
    all_err = []
    all_y0 = []
    all_c = []
    for i in range(len(decision_makers)):
        loss = decision_makers[i].compute_decision_loss(y.flatten())
        err = decision_makers[i].compute_decision_gap(y.flatten())
        total_loss += loss
        total_err += err
        all_loss.append(loss)
        all_err.append(err)
        all_y0.append(decision_makers[i].y0)
        all_c.append(decision_makers[i].c)
    total_loss /= len(decision_makers)
    total_err /= len(decision_makers)
    return (
        total_err,
        total_loss,
        torch.tensor(all_err),
        torch.tensor(all_loss),
        torch.tensor(all_y0),
        torch.tensor(all_c),
    )
