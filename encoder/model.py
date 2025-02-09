import json
import numpy as np
import torch
import torch.optim
import tcnn
import utils

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

CUDA = torch.cuda.is_available()
CHANNELS = 9
EPOCHS = 100

def same_seed(seed): 
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def collate(x):
    trees = []
    targets = []

    for tree, target in x:
        trees.append(tree)
        targets.append(target)

    targets = torch.tensor(targets)
    return trees, targets

class TCNNRegression:

    def __init__(self):
        self.trained = False

    def fit(self, X, Y):
        if isinstance(Y, list):
            Y = np.array(Y).reshape(-1, 1).astype(np.float32)
        pairs = list(zip(X, Y))
        dataset = DataLoader(pairs, batch_size=32, 
                            shuffle=True, collate_fn=collate)
        
        self._net = tcnn.TreeNet(CHANNELS)
        self._net.train()
        optimizer = torch.optim.Adam(self._net.parameters())
        loss_fn = torch.nn.MSELoss()

        losses = []
        for epoch in range(1, 1+EPOCHS):
            accum = 0.
            for x, y in dataset:
                y_pred = self._net(x)
                loss = loss_fn(y_pred, y)
                accum += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            accum /= len(dataset)
            losses.append(accum)
            print("epoch {} loss {}".format(epoch, accum))
        self.trained = True

    def predict(self, X):
        assert self.trained
        self._net.eval()
        pred = self._net(X).cpu().detach().numpy()
        return pred


if __name__ == "__main__":
    same_seed(1145141101)
    X, Y = [], []
    SPLIT_POS = 4000
    with open("../data/plan_test/imdb_data") as f:
        lines = f.readlines()
        for l in lines:
            items = l.split('\t')
            plan = utils.generate_plan_tree(json.loads(items[1])['plan'])
            latency = float(items[2])
            X.append(plan)
            Y.append(latency)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = TCNNRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    print("mean squared error: {}".format(mse))
    print("mean absolute error: {}".format(mae))

    # depict illustration
    fig, ax = plt.subplots()
    x = np.linspace(0, 100, 10)
    y = x

    ax.plot(x, y, label='x=y', color='orange')

    ax.set_title('Diagonal Line in Subplot')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal', adjustable='box')
    ax.scatter(y_test, preds, color='tab:blue', s=3)
    ax.set_ylabel("Predictions.")
    ax.set_xlabel("Ground Truth.")
    ax.set_title("Predictions vs. Ground Truth (seconds)")
    plt.savefig("test.png")
