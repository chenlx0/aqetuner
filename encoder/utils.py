import numpy as np
import torch
import requests
import time
HOST = 'http://localhost:6725'


def exec_analyze_query(sql: str, db: str, settings: dict):
    sql = "explain analyze json=1 " + sql
    settings['database'] = db
    time1 = time.time()
    r = requests.post(HOST, data=sql, params=settings)
    time2 = time.time()
    r.encoding = 'utf-8'
    resp = r.json()
    return resp['plan'], time2-time1

NodeTypes = [
    "Projection",
    "MergingAggregated",
    "Exchange",
    "Aggregating",
    "Join",
    "Filter",
    "TableScan"
]

RowsNorm = 1e7
CostNorm = 1e8
TimeNorm = 1e4

def plan_tree_to_vecs(plan):
    res = []
    nodes= [plan]
    while len(nodes) > 0:
        node = nodes.pop()
        arr = [0. for _ in range(9)]
        stats = node['Statistic']
        arr[len(NodeTypes)] = 0. if 'RowCount' not in stats else stats['RowCount'] / RowsNorm
        arr[len(NodeTypes) + 1] = 0. if 'Cost' not in stats else stats['Cost'] / CostNorm
        res.append(arr)
        if 'Children' in node:
            nodes = node['Children'] + nodes
    assert len(res) <= 100
    if len(res) < 100:
        for _ in range(100-len(res)):
            res.append([0. for _ in range(9)])
    return res 


def generate_plan_tree(plan):
    arr = [0. for _ in range(9)] # type one-hot + rows + cost
    if plan['NodeType'] in NodeTypes:
        arr[NodeTypes.index(plan['NodeType'])] = 1.
    stats = plan['Statistic']
    arr[len(NodeTypes)] = 0. if 'RowCount' not in stats else stats['RowCount'] / RowsNorm
    arr[len(NodeTypes) + 1] = 0. if 'Cost' not in stats else stats['Cost'] / CostNorm

    items = [arr]
    if 'Children' in plan:
        for next in plan['Children']:
            items.append(generate_plan_tree(next))
    while len(items) < 3:
        items.append([[0. for _ in range(9)]])
    return items


def generate_latency(plan):
    profiles = plan['Profiles']
    arr = [0.] if 'WallTimeMs' not in profiles else [profiles['WallTimeMs'] / TimeNorm]
    items = [arr]
    if 'Children' in plan:
        for next in plan['Children']:
            items.append(generate_latency(next))
    while len(items) < 3:
        items.append([[0.]])
    return items


class TreeConvolutionError(Exception):
    pass

def _is_leaf(x, left_child, right_child):
    has_left = left_child(x) is not None
    has_right = right_child(x) is not None
    
    if has_left != has_right:
        raise TreeConvolutionError(
            "All nodes must have both a left and a right child or no children"
        )

    return not has_left

def _flatten(root, transformer, left_child, right_child):
    """ turns a tree into a flattened vector, preorder """

    if not callable(transformer):
        raise TreeConvolutionError(
            "Transformer must be a function mapping a tree node to a vector"
        )

    if not callable(left_child) or not callable(right_child):
        raise TreeConvolutionError(
            "left_child and right_child must be a function mapping a "
            + "tree node to its child, or None"
        )

    accum = []

    def recurse(x):
        if _is_leaf(x, left_child, right_child):
            accum.append(transformer(x))
            return

        accum.append(transformer(x))
        recurse(left_child(x))
        recurse(right_child(x))

    recurse(root)

    try:
        accum = [np.zeros(len(accum[0]))] + accum
    except:
        raise TreeConvolutionError(
            "Output of transformer must have a .shape (e.g., numpy array)"
        )
    
    return np.array(accum)

def _preorder_indexes(root, left_child, right_child, idx=1):
    """ transforms a tree into a tree of preorder indexes """
    
    if not callable(left_child) or not callable(right_child):
        raise TreeConvolutionError(
            "left_child and right_child must be a function mapping a " +
            "tree node to its child, or None"
        )


    if _is_leaf(root, left_child, right_child):
        # leaf
        return idx

    def rightmost(tree):
        if isinstance(tree, tuple):
            return rightmost(tree[2])
        return tree
    
    left_subtree = _preorder_indexes(left_child(root), left_child, right_child,
                                     idx=idx+1)
    
    max_index_in_left = rightmost(left_subtree)
    right_subtree = _preorder_indexes(right_child(root), left_child, right_child,
                                      idx=max_index_in_left + 1)

    return (idx, left_subtree, right_subtree)
    
def _tree_conv_indexes(root, left_child, right_child):
    """ 
    Create indexes that, when used as indexes into the output of `flatten`,
    create an array such that a stride-3 1D convolution is the same as a
    tree convolution.
    """
    
    if not callable(left_child) or not callable(right_child):
        raise TreeConvolutionError(
            "left_child and right_child must be a function mapping a "
            + "tree node to its child, or None"
        )
    
    index_tree = _preorder_indexes(root, left_child, right_child)

    def recurse(root):
        if isinstance(root, tuple):
            my_id = root[0]
            left_id = root[1][0] if isinstance(root[1], tuple) else root[1]
            right_id = root[2][0] if isinstance(root[2], tuple) else root[2]
            yield [my_id, left_id, right_id]
                                           
            yield from recurse(root[1])
            yield from recurse(root[2])
        else:
            yield [root, 0, 0]

    return np.array(list(recurse(index_tree))).flatten().reshape(-1, 1)

def _pad_and_combine(x):
    assert len(x) >= 1
    assert len(x[0].shape) == 2

    for itm in x:
        if itm.dtype == np.dtype("object"):
            raise TreeConvolutionError(
                "Transformer outputs could not be unified into an array. "
                + "Are they all the same size?"
            )
    
    second_dim = x[0].shape[1]
    for itm in x[1:]:
        assert itm.shape[1] == second_dim

    max_first_dim = max(arr.shape[0] for arr in x)

    vecs = []
    for arr in x:
        padded = np.zeros((max_first_dim, second_dim))
        padded[0:arr.shape[0]] = arr
        vecs.append(padded)

    return np.array(vecs)

def prepare_trees(trees, transformer, left_child, right_child, cuda=False):
    flat_trees = [_flatten(x, transformer, left_child, right_child) for x in trees]
    flat_trees = _pad_and_combine(flat_trees)
    flat_trees = torch.Tensor(flat_trees)

    # flat trees is now batch x max tree nodes x channels
    flat_trees = flat_trees.transpose(1, 2)
    if cuda:
        flat_trees = flat_trees.cuda()

    indexes = [_tree_conv_indexes(x, left_child, right_child) for x in trees]
    indexes = _pad_and_combine(indexes)
    indexes = torch.Tensor(indexes).long()

    if cuda:
        indexes = indexes.cuda()

    return (flat_trees, indexes)


def flatten_tree(tree, flat_list):
    flatten_list(tree, flat_list)
    return flat_list

def flatten_list(lst, flat_list):
    for item in lst:
        if isinstance(item, list) and len(item) != 9:
            flatten_list(item, flat_list)
        else:
            flat_list.append(item)