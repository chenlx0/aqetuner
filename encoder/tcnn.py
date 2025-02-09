import torch
import json
import torch.nn as nn
from utils import prepare_trees, generate_plan_tree

def left_child(x):
    if len(x) != 3:
        return None
    return x[1]

def right_child(x):
    if len(x) != 3:
        return None
    return x[2]

def features(x):
    return x[0]

class BinaryTreeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BinaryTreeConv, self).__init__()

        self.__in_channels = in_channels
        self.__out_channels = out_channels
        # we can think of the tree conv as a single dense layer
        # that we "drag" across the tree.
        self.weights = nn.Conv1d(in_channels, out_channels, stride=3, kernel_size=3)

    def forward(self, flat_data):
        trees, idxes = flat_data
        orig_idxes = idxes
        idxes = idxes.expand(-1, -1, self.__in_channels).transpose(1, 2)
        expanded = torch.gather(trees, 2, idxes)

        results = self.weights(expanded)

        # add a zero vector back on
        zero_vec = torch.zeros((trees.shape[0], self.__out_channels)).unsqueeze(2)
        zero_vec = zero_vec.to(results.device)
        results = torch.cat((zero_vec, results), dim=2)
        return (results, orig_idxes)

class TreeActivation(nn.Module):
    def __init__(self, activation):
        super(TreeActivation, self).__init__()
        self.activation = activation

    def forward(self, x):
        return (self.activation(x[0]), x[1])

class TreeLayerNorm(nn.Module):
    def forward(self, x):
        data, idxes = x
        mean = torch.mean(data, dim=(1, 2)).unsqueeze(1).unsqueeze(1)
        std = torch.std(data, dim=(1, 2)).unsqueeze(1).unsqueeze(1)
        normd = (data - mean) / (std + 0.00001)
        return (normd, idxes)
    
class DynamicPooling(nn.Module):
    def forward(self, x):
        return torch.max(x[0], dim=2).values


class TreeNet(nn.Module):
    def __init__(self, in_channels, output=8):
        super(TreeNet, self).__init__()
        self.__in_channels = in_channels
        self.__cuda = False

        self.tree_conv = nn.Sequential(
            BinaryTreeConv(self.__in_channels, 256),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(256, 128),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(128, 64),
            TreeLayerNorm(),
            DynamicPooling(),
            nn.Linear(64, output),
        )

    def in_channels(self):
        return self.__in_channels
        
    def forward(self, x):
        trees = prepare_trees(x, features, left_child, right_child,
                              cuda=self.__cuda)
        return self.tree_conv(trees)

    def cuda(self):
        self.__cuda = True
        return super().cuda()

class TCNNEncoder(nn.Module):
    def __init__(self, plan_in_channel, knobs_dim, output=32):
        super(TCNNEncoder, self).__init__()
        self.channel = plan_in_channel
        self.knobs_dim = knobs_dim
        self._net = TreeNet(plan_in_channel)
        self.m = torch.nn.Linear(11, output)

    def forward(self, plan, knobs):
        x = self._net(plan)
        tmp = torch.cat((x, torch.Tensor(knobs)), 1)
        return self.m(tmp)

if __name__ == "__main__":
    encoder = TCNNEncoder(9, 3)
    with open("test_plan") as f:
        plan = eval(f.read())
        plan = generate_plan_tree(plan)
        print(plan)
    vec = encoder([plan, plan], [[0.5, 0.4, 0.6], [0.2, 0.4, 0.5]])
    print(vec)
