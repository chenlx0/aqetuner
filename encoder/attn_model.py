import json
import numpy as np
import torch
import torch.optim
import pickle

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from attn_encoder import AttrEncoder

from torch.utils.data import DataLoader

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
    knobs = []
    targets = []

    for tree, knob, target in x:
        trees.append(tree)
        knobs.append(knob)
        targets.append(target)

    targets = torch.tensor(targets)
    return trees, knobs, targets

class TCNNRegression:

    def __init__(self):
        self.trained = False

    def fit(self, X, K, Y):
        if isinstance(Y, list):
            Y = np.array(Y).reshape(-1, 1).astype(np.float32)
        pairs = list(zip(X, K, Y))
        dataset = DataLoader(pairs, batch_size=32, 
                            shuffle=True, collate_fn=collate)
        self._net = AttrEncoder(CHANNELS, 11)
        self._net.train()
        optimizer = torch.optim.Adam(self._net.parameters())
        loss_fn = torch.nn.MSELoss()

        losses = []
        for epoch in range(1, 1+EPOCHS):
            accum = 0.
            for x, k, y in dataset:
                y_pred = self._net(x, k)
                # y_pred = y_pred.view(1, -1).reshape(1, -1)
                loss = loss_fn(y_pred, y)
                accum += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            accum /= len(dataset)
            losses.append(accum)
            print("epoch {} loss {}".format(epoch, accum))
        self.trained = True

    def predict(self, X, K):
        assert self.trained
        self._net.eval()
        pred = self._net(X, K)
        # pred = self._net(X).cpu().detach().numpy()
        return pred


if __name__ == "__main__":
    same_seed(1145141101)
    X, K, Y = [], [], []
    SPLIT_POS = 4000
    # with open("../dataset/imdb_data.bak") as f:
    #     lines = f.readlines()
    #     for l in lines:
    #         items = l.split('\t')
    #         plan = utils.generate_plan_tree(json.loads(items[1])['plan'])
    #         latency = float(items[2])
    #         X.append(plan)
    #         Y.append(latency)
    with open("../dataset/sample_test_0618") as f:
        data = f.readlines()
        for d in data:
            d_list = d.split('\t')
            X.append(json.loads(d_list[0])['plan'])
            K.append(eval(d_list[1]))
            Y.append(json.loads(d_list[2])['elapsed'])
    X_train, X_test, K_train, K_test, y_train, y_test = train_test_split(X, K, Y, test_size=0.2, random_state=42)
    model = TCNNRegression()
    model.fit(X_train, K_train, y_train)
    fw = open("attn_model.pkl", "wb")
    pickle.dump(model, fw)
    # with open('attn_model.pkl', 'rb') as file:
    #     model = pickle.load(file)
    preds = model.predict(X_test, K_test).detach().numpy().flatten()
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    print("mean squared error: {}".format(mse))
    print("mean absolute error: {}".format(mae))
