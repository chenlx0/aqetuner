import numpy as np
import xgboost

import shap
import json

RowsNorm = 1e7
CostNorm = 1e8
TimeNorm = 1e4

NodeTypes = [
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
    "Buffer"
]

def get_analyzed_data(path):
    xs, ys = {}, {}
    for tp in NodeTypes:
        xs[tp], ys[tp] = [], []
    with open(path) as f:
        lines = f.read().splitlines()
    for l in lines:
        items = l.split('\t')
        root = json.loads(items[0])['plan']
        stack = [root]
        while len(stack) > 0:
            next = stack.pop()
            arr = eval(items[2])
            arr += [next['Statistic']['RowCount']/RowsNorm, next['Statistic']['Cost']/CostNorm]
            xs[next['NodeType']].append(arr)
            ys[next['NodeType']].append(next['Profiles']['OutputWaitTimeMs']/1e3)
            if 'Children' in next:
                stack = next['Children'] + stack
    return xs, ys

def main(target_file):
    xs, ys = get_analyzed_data("data/stats_samples")
    f = open(target_file, 'w+')
    for tp in NodeTypes:
        if len(xs[tp]) > 0:
            X = np.array(xs[tp])
            y = np.reshape(np.array(ys[tp]), (-1,1))
            Xd = xgboost.DMatrix(X, label=y)
            model = xgboost.train({"eta": 1, "max_depth": 5, "base_score": 0, "lambda": 0}, Xd, 1)
            pred = model.predict(Xd, output_margin=True)
            explainer = shap.TreeExplainer(model)
            explanation = explainer(Xd)
            shap_values = np.abs(explanation.values)
            shap_values = shap_values.mean(0)
            print(tp)
            f.write(tp + ' ')
            for i, v in enumerate(shap_values.tolist()):
                if v > 0.:
                    f.write(str(i) + ' ')
            f.write('\n')
    f.close()

if __name__ == "__main__":
    main('data/correlation.txt')
