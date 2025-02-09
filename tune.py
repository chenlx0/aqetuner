
import torch as t
import numpy as np
import json
import copy
import random

from scipy.stats import norm, qmc
from connector import *
from network import LatentModel
# from encoder.utils import generate_plan_tree, plan_tree_to_vecs
from pair_encoder import TPair

DEVICE = 'cpu'
DB = 'stats'

def eic(mu, stdvar, best, pr):
    z = (best - mu) / stdvar
    return pr * ((best - mu) * norm.cdf(z) + stdvar * norm.pdf(z))

def initialize_model(model_path):
    state_dict = t.load(model_path)
    model = LatentModel(32).to(DEVICE)
    model.load_state_dict(state_dict['model'])
    model.train(False)
    return model

def initialize_data(workload_path):
    with open(workload_path) as f:
        queries = f.read().splitlines()
    plans = []
    for q in queries:
        actual_db = DB if DB != 'tpch' else 'tpch_100'
        plans.append(get_query_plan(q, actual_db)['plan'])
    return queries, plans

def initialize_observed_data(path):
    pairs, elapsed = [], []
    with open(path) as f:
        line = f.readline()
        while line:
            items = line.split('\t')
            stats = json.loads(items[3])
            if len(stats) > 0:
                pairs.append(TPair(json.loads(items[1])['plan'], json.loads(items[2])))
                elapsed.append([stats['elapsed'], stats['fail']])
            line = f.readline()
    return pairs, elapsed

def get_configuration(path):
    with open(path) as f:
        conf = json.load(f)
    return conf 

def denormalize_knobs(conf: dict, values: list) -> dict:
    res = {}
    for i, key in enumerate(list(conf.keys())):
        detail = conf[key]
        if detail['type'] == 'int/continuous':
            para = values[i]
            res[key] = round(detail['lower'] + (detail['upper'] - detail['lower']) * para)
        else:
            idx = round(para)
            res[key] = detail['value'][idx]
    return res

def main_np():
    queries, plans = initialize_data(f"workloads/{DB}")
    model = initialize_model(f"checkpoint/{DB}.pth.tar")
    model.train(False)
    obs_pairs, obs_elapseds = initialize_observed_data(f"data/{DB}_samples")
    iters = 30
    samples = 2**10
    conf = get_configuration('knobs.json')
    best_elapsed = [100. for _ in range(len(queries))]
    for iter in range(iters):
        elapsed = []
        selected_confs = []
        for i in range(len(queries)):
            # recommend configuration
            start = time.time()
            candidate_pairs = [TPair(plans[i], t.rand(len(conf)).tolist()) for _ in range(samples)]
            preds_l, preds_e, _ = model(obs_pairs, t.Tensor(obs_elapseds), candidate_pairs)
            latency = preds_l.mean
            fail = preds_e.mean
            cond = fail > 0.5
            cond = cond.all(1)
            latency[cond] = 1000.


            idx = latency.argmin()
            next_conf = denormalize_knobs(conf, candidate_pairs[idx]._knobs)
            print(f"inference cost {time.time() - start}")
            # print(next_conf)
            # evaluate conf
            actual_db = DB if DB != "tpch" else "tpch_100"
            stats = exec_query(queries[i], actual_db, next_conf)
            if len(stats) == 0:
                print('bad params')
                stats['elapsed'] = 100.
            print("query %d elapsed %f fail %d predict mean %f predict fail %f" % (i, stats['elapsed'], stats['fail'], latency[idx], fail[idx]))

            obs_pairs.append(candidate_pairs[idx])
            obs_elapseds.append([stats['elapsed'], stats['fail']])
            if stats['fail'] == 0:
                best_elapsed[i] = min(stats['elapsed'], best_elapsed[i])
        
        # collect data
        # if iter == 0:
        #     obs_plan.clear()
        #     obs_conf.clear()
        #     obs_elapsed.clear()

        print("best scores %f" % sum(best_elapsed))
        sorted_elapsed = copy.deepcopy(best_elapsed)
        sorted_elapsed.sort()

        print(sorted_elapsed)


if __name__ == "__main__":
    main_np()
        
