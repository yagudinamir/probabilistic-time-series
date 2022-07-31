import torch
from torch.nn import Softplus

nll_criterion = lambda mu, sigma, y: torch.log(sigma) / 2 + ((y - mu) ** 2) / (2 * sigma)


def train_model(net, optimizer):
    eps = 0.01 * 8
    alpha = 0.5
    running_loss = []
    for epoch in range(40):
        epoch_loss = 0
        for x, y in zip(inputs, labels):

            x = torch.tensor([x], dtype=torch.float, requires_grad=True)
            y = torch.tensor([y], dtype=torch.float)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = net(x)
            mu, sig = output[0], Softplus(output[1]) + 1e-6
            loss = nll_criterion(mu, sig, y)
            loss.backward(retain_graph=True)

            x_a = x + eps * (torch.sign(x.grad.data))
            optimizer.zero_grad()

            output_a = net(x_a)
            mu_a, sig_a = output_a[0], Softplus(output_a[1]) + 1e-6

            loss = alpha * nll_criterion(mu, sig, y) + (1 - alpha) * nll_criterion(mu_a, sig_a, y)
            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        running_loss.append(epoch_loss / len(inputs))


n_models = 5

ensemble = []

for i in range(n_models):
    model = train_model()
    ensemble.append(model)
