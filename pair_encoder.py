
import torch as t
from torch import nn
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from tree_utils import generate_plan_tree, TreeNet

import os


HIDDEN_DIM = 32
MAX_NODES = 200
DB = os.getenv('DB')

RowsNorm = 1e7

CKNodeTypes = [
    "Projection",
    "MergingAggregated",
    "Exchange",
    "Aggregating",
    "Join",
    "Filter",
    "TableScan",
    "Limit",
    "Sorting",
    "CTERef",
    "Buffer",
    "Union",
    "EnforceSingleRow",
    "Window",
    "Values",
    "PartitionTopN",
    ""
]

COMPARATORS = [
    ">",
    "<",
    "=",
    ">=",
    "<=",
    "!=",
    "LIKE",
    "IN",
    "NOT LIKE"
]

NODE_DIM = 256
KNOB_DIM = 16

COL_NAME, MIN_MAX = [], []

class TPair(object):

    def __init__(self, json_plan, knobs):
        # self.tree_plan = generate_plan_tree(json_plan)
        self.NodeTypes = CKNodeTypes
        self._knobs = knobs
        self._parse_plan(json_plan)
        self.Knobs = t.eye(KNOB_DIM)
        for i, v in enumerate(knobs):
            if i < KNOB_DIM:
                self.Knobs[i][i] = v
        for i in range(KNOB_DIM - len(knobs)):
            knobs.append(0.)
        self.plat_knobs = t.Tensor(knobs)

    def _node_to_vec(self, node):
        if 'NodeType' not in node:
            node['NodeType'] = ''
        vec_len = len(CKNodeTypes) + 2
        arr = [0. for _ in range(NODE_DIM)]
        stats = {} if 'Statistic' not in node else node['Statistic']
        arr[vec_len-1] = 0. if 'RowCount' not in stats else stats['RowCount'] / RowsNorm
        arr[vec_len-2] = node['depth']
        arr[self.NodeTypes.index(node['NodeType'])] = 1.

        # concat other 1-hot encoding
        if node['NodeType'] == 'TableScan' and 'Where' in node:
            emb = encode_predicate(node['Where'])
        elif node['NodeType'] == 'Join' and 'Condition' in node:
            emb = encode_join(node['Condition'])
        elif node['NodeType'] == 'Aggregating' and 'GroupByKeys' in node:
            emb = encode_aggregate(node['GroupByKeys'])
        else:
            return arr
        
        arr[vec_len:] = emb[:NODE_DIM-vec_len]
        while len(arr) < NODE_DIM:
            arr.append(0.)
        return arr

    def _parse_plan(self, root):
        vec_len = NODE_DIM
        nodes = [root]
        res = []
        mask = t.ones(MAX_NODES, MAX_NODES, dtype=bool)
        root['depth'] = 1.

        vis = []
        while len(nodes) > 0:
            next = nodes.pop()
            arr = self._node_to_vec(next)
            id = len(res)
            res.append(arr)
            if 'parent_id' in next:
                mask[next['parent_id']][id] = False
                mask[id][next['parent_id']] = False
                mask[id][id] = False
            if 'Children' in next:
                for p in next['Children']: 
                    if p['NodeId'] not in vis:
                        vis.append(p['NodeId'])
                        p['parent_id'] = id 
                        p['depth'] = next['depth'] + 1
                        nodes.append(p)
        
        for i in range(MAX_NODES - len(res)):
            res.append([0. for _ in range(vec_len)])
        
        self.Vecs = t.Tensor(res)
        self.Mask1 = mask

    def _spectral_encoding(self, k):
        # compute indegree matrix according to Mask1
        indegree = t.zeros(MAX_NODES, MAX_NODES)
        for i in range(MAX_NODES):
            count = 0
            for j in range(MAX_NODES):
                if self.Mask1[j][i] == t.tensor(True):
                    count = count + 1
            indegree[i][i] = count
        # compute Laplacian matrix
        L = indegree - self.Mask1.int()
        # compute eigenvalues and eigenvectors
        laplacian_matrix = csgraph.laplacian(L.numpy(), normed=False)
        eigenvalues, eigenvectors = eigsh(laplacian_matrix, k=k, which='SM')  # 'SM' 表示最小特征值
        # eigenvectors.shape: MAX_NODES x k
        pass

class FFN(nn.Module):

    def __init__(self, input=16, output=16):
        super(FFN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input, output),
            nn.ReLU(),
            nn.Linear(output, output)
        )

    def forward(self, vecs):
        return self.layers(vecs)

class AttnEncoder(nn.Module):

    def __init__(self, node_dim=NODE_DIM, knob_dim=KNOB_DIM, heads=4):
        super(AttnEncoder, self).__init__()
        self._node_dim = node_dim
        self._knob_dim = knob_dim
        self._heads = heads
        self.nodes_attn = nn.MultiheadAttention(node_dim, heads, batch_first=True)
        self.ffn1 = FFN(node_dim, HIDDEN_DIM)
        self.norm1 = nn.LayerNorm(HIDDEN_DIM)
        self.knobs_attn = nn.MultiheadAttention(knob_dim, heads, batch_first=True)
        self.ffn2 = FFN(knob_dim, HIDDEN_DIM)
        self.norm2 = nn.LayerNorm(HIDDEN_DIM)
        self.cross_attn = nn.MultiheadAttention(HIDDEN_DIM, heads, batch_first=True)
        self.norm3 = nn.LayerNorm(HIDDEN_DIM)

        # self.test_linear = nn.Linear(node_dim + knob_dim, HIDDEN_DIM)

    def forward(self, pairs: list):
        # nodes = batch * max_nodes * node_dim
        # knobs = batch * max_knobs * knob_dim
        # mask1 = batch * 100 * 100, mask2 = batch * 16 * 16
        nodes = t.stack([p.Vecs for p in pairs])
        knobs = t.stack([p.Knobs for p in pairs])
        mask1 = t.stack([p.Mask1 for _ in range(self._heads) for p in pairs])
        x, weights = self.nodes_attn(nodes, nodes, nodes, attn_mask=mask1)
        x = t.nan_to_num(x)
        node_vecs = self.norm1(self.ffn1(x))
        # return self.test_linear(t.cat([x.sum(dim=1), knobs.sum(dim=1)], dim=1))
        x, weights = self.knobs_attn(knobs, knobs, knobs)
        x = t.nan_to_num(x)
        knob_vecs = self.norm2(self.ffn2(x))
        x, _ = self.cross_attn(node_vecs, knob_vecs, knob_vecs)
        x = t.max(x, dim=1).values
        return self.norm3(x)


class TreeEncoder(nn.Module):

    def __init__(self, node_dim=NODE_DIM, knob_dim=KNOB_DIM):
        super(TreeEncoder, self).__init__()
        self.treenet = TreeNet(node_dim, HIDDEN_DIM - knob_dim)

    def forward(self, pairs: list):
        # nodes = batch * max_nodes * node_dim
        # knobs = batch * max_knobs * knob_dim
        # mask1 = batch * 100 * 100, mask2 = batch * 16 * 16
        trees = [p.tree_plan for p in pairs]
        knobs = t.stack([p.plat_knobs for p in pairs])
        tree_vecs = self.treenet(trees)
        x = t.cat([tree_vecs, knobs], dim=1)
        return x


def load_column_data(fname):
    column_name, min_max_vals = [], []
    with open(f"statistics/{fname}") as f:
        lines = f.readlines()
    for line in lines:
        items = line.split()
        column_name.append(items[1])
        if len(items) == 4:
            min_max_vals.append((int(items[2]), int(items[3])))
        else:
            min_max_vals.append(())
    return column_name, min_max_vals


def encode_predicate(s: str):
    global COL_NAME, MIN_MAX
    if len(COL_NAME) == 0:
        COL_NAME, MIN_MAX = load_column_data(DB)
    res = []
    predicates = []
    flag = 0
    p = ""
    for c in s:
        if c == '(':
            p = "" if flag == 0 else p
            flag += 1
        elif c == ')':
            if flag == 1:
                predicates.append(p)
            flag -= 1
        elif c == ',':
            pass
        elif flag > 0:
            p += c
    if len(predicates) == 0 and 'Filters' not in s:
        predicates.append(s)
    
    for predicate in predicates:
        items = predicate.split()
        try:
            if items[1] == 'NOT':
                items[1] = 'NOT LIKE'
                items.remove('LIKE')
            emb = [0. for _ in range(len(COL_NAME) + len(COMPARATORS) + 6)]
            emb[COL_NAME.index(items[0])] = 1.
            emb[len(COL_NAME) + COMPARATORS.index(items[1])] = 1.
            vals = MIN_MAX[COL_NAME.index(items[0])]
            if len(vals) == 2:
                emb[len(COL_NAME) + len(COMPARATORS)] = \
                    (int(items[2]) - vals[0]) / (vals[1] - vals[0])
            res += emb
        except Exception as e:
            pass
    return res

def encode_join(conds: list):
    global COL_NAME, MIN_MAX
    if len(COL_NAME) == 0:
        COL_NAME, MIN_MAX = load_column_data(DB)
    emb = [0. for _ in range(len(COL_NAME) + len(COMPARATORS) + 6)]
    for s in conds:
        tbls = s.split()
        for tb in tbls:
            if tb in COL_NAME:
                emb[COL_NAME.index(tb)] = 1.
    return emb

def encode_aggregate(keys: list):
    global COL_NAME, MIN_MAX
    if len(COL_NAME) == 0:
        COL_NAME, MIN_MAX = load_column_data(DB)
    emb = [0. for _ in range(len(COL_NAME) + len(COMPARATORS) + 6)]
    for tb in keys:
        if tb in COL_NAME:
            emb[COL_NAME.index(tb)] = 1.
    return emb


if __name__ == "__main__":
    # with open("encoder/test_plan") as f:
    #     c = json.loads(f.read())
    # p = TPair(c, [0.1, 0.3, 0.4, 0.34, 0.5])
    # encoder = TreeEncoder(16, 16)
    # print(encoder([p, p, p]))
    x = encode_predicate("(s_comment LIKE '%Customer%') AND (s_comment LIKE '%Complaints%')")
    print(x)
    x = encode_join(["o_custkey == c_custkey"] )
    print(x)

