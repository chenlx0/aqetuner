
import torch as t
import numpy as np
import argparse
import json
import copy
import random
import os

from scipy.stats import norm, qmc
from connector import *
from network import LatentModel
# from encoder.utils import generate_plan_tree, plan_tree_to_vecs
from pair_encoder import TPair

DEVICE = 'cuda' if t.cuda.is_available() else "cpu"
DB = os.getenv('DB')

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
        actual_db = DB
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

def main_np(sqls, sample_file, model_file, knob_file, iters, output_file):
    queries, plans = initialize_data(sqls)
    model = initialize_model(model_file)
    model.train(False)
    obs_pairs, obs_elapseds = initialize_observed_data(sample_file)
    samples = 2**10
    conf = get_configuration(knob_file)
    best_elapsed = [100. for _ in range(len(queries))]
    for iter in range(iters):
        elapsed = []
        selected_confs = []
        for i in range(len(queries)):
            # recommend configuration
            candidate_pairs = [TPair(plans[i], t.rand(len(conf)).tolist()) for _ in range(samples)]
            preds_l, preds_e, _ = model(obs_pairs, t.Tensor(obs_elapseds), candidate_pairs)
            latency = preds_l.mean
            fail = preds_e.mean
            cond = fail > 0.5
            cond = cond.all(1)
            latency[cond] = 1000.


            idx = latency.argmin()
            next_conf = denormalize_knobs(conf, candidate_pairs[idx]._knobs)
            # print(next_conf)
            # evaluate conf
            actual_db = DB
            stats = exec_query(queries[i], actual_db, next_conf)
            if len(stats) == 0:
                print('bad params')
                stats['elapsed'] = 100.
            print("query %d elapsed %f fail %d predict mean %f predict fail %f" % (i, stats['elapsed'], stats['fail'], latency[idx], fail[idx]))

            obs_pairs.append(candidate_pairs[idx])
            obs_elapseds.append([stats['elapsed'], stats['fail']])
            if stats['fail'] == 0:
                best_elapsed[i] = min(stats['elapsed'], best_elapsed[i])
        with open(output_file, "w+") as fw:
            fw.writelines([str(sum(best_elapsed))])
        
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
    parser = argparse.ArgumentParser(description='Configuration Tuning')
    parser.add_argument('--knob_file', type=str, required=True,
                        help='specifies the path of knob file')
    parser.add_argument('--sample_file', type=str, required=True,
                        help='specify the path of file that contains sampled data')
    parser.add_argument('--db', type=str, required=True,
                        help='specifies the database')
    parser.add_argument('--sqls', type=str, required=True,
                        help='specifies the path of SQL workloads')
    parser.add_argument('--model', type=str, required=True,
                        help='specifies the path of model file, corresponding to --model_output in training phase')
    parser.add_argument('--max_iteration', type=int, required=True,
                        help='specifies the maximum number of iterations for tuning')
    parser.add_argument('--result_file', type=str, required=True,
                        help='specifies the file to save the tuning result')

    args = parser.parse_args()
    DB = args.db
    main_np(args.sqls, args.sample_file, args.model, args.knob_file, args.max_iteration, args.result_file)
        
