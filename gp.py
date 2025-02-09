import torch as t
from torch import nn
import gpytorch
import json

from encoder.tcnn import TCNNEncoder
from encoder.utils import plan_tree_to_vecs

class LargeFeatureExtractor(t.nn.Sequential):
    def __init__(self, input_dim):
        super(LargeFeatureExtractor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256,256),
            nn.LeakyReLU(),
            nn.Linear(256,128),
            nn.LeakyReLU(),
            nn.Linear(128, 16),
        )

    def forward(self, x):
        ''' Given input of size (batch_size x input_dim), compute output of the network '''
        return self.net(x)


class QueryLevelGP(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood):
        super(QueryLevelGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module =  gpytorch.kernels.RBFKernel()
        self.encoder = LargeFeatureExtractor(train_x.size(-1))
        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

    def forward(self, x):
        x = self.encoder(x)
        x = self.scale_to_bounds(x)
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

# class QueryLevelRegression(t.nn.Module):

#     def __init__(self, plans, confs, elapseds):
#         super(QueryLevelRegression, self).__init__()
#         self.encoder = TCNNEncoder(9, 3, 8)
#         train_x = self.encoder(plans, confs)
#         self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
#         self.likelihood.train()
#         self.gp = QueryLevelGP(train_x, elapseds, self.likelihood)

#     def forward(self, plans, confs):
#         x = self.encoder(plans, confs)
#         return self.gp(x)

def load_observed_data(path):
    plan, conf, elapsed = [], [], []
    with open(path) as f:
        line = f.readline()
        while line:
            items = line.split('\t')
            stats = json.loads(items[3])
            if len(stats) > 0 and stats['elapsed'] < 50:
                plan.append(plan_tree_to_vecs(json.loads(items[1])['plan']))
                conf.append(json.loads(items[2]))
                elapsed.append(stats['elapsed'])
            line = f.readline()
    elapsed = t.Tensor(elapsed)
    return plan, conf, elapsed

def project_x(plans, confs):
    plans = t.Tensor(plans).flatten(start_dim=1)
    confs = t.Tensor(confs)
    train_x = t.cat([plans, confs], dim=1)
    return train_x

def gp_train(plans, confs, elapseds):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    # plans, confs, elapseds = load_observed_data("data/job-samples")
    train_x = project_x(plans, confs)
    model = QueryLevelGP(train_x, elapseds, likelihood)

    likelihood.train()
    model.train()

    optimizer = t.optim.Adam([
        {'params': model.encoder.parameters()},
        {'params': model.covar_module.parameters()},
        {'params': model.mean_module.parameters()},
        {'params': model.likelihood.parameters()},
    ], lr=0.01)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(500):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, elapseds)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (
            i + 1, 400, loss.item()
        ))
        optimizer.step()

    return model

if __name__ == "__main__":
    plans, confs, elapseds = load_observed_data("data/job-samples")
    model =  gp_train(plans, confs, elapseds)
    model.eval()
    y = model(project_x(plans, confs))
    print(y.mean, y.stddev)
    print(elapseds)
