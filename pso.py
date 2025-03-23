

import threading
import queue
import json
import time
import argparse
import math
import time

import numpy as np

from scipy.stats import qmc
from connector import exec_query, get_query_plan


class SinglePSOSampler(threading.Thread):

    def __init__(self, sql, db, knobs: dict, p_num, res_q: queue.Queue, arg_q: queue.Queue, 
                 is_consumer, records: list, max_tries=9, *args, **kwargs):
        self._sql = sql
        self._db = db
        self._knobs = knobs
        self._res_q = res_q
        self._arg_q = arg_q
        self._p_num = p_num
        self._is_consumer = is_consumer
        self._records = records
        self._max_tries = max_tries
        super().__init__(*args, **kwargs)

    def produce(self):
        metrics = ['elapsed']
        global_best = [10000., 10000.,]
        global_best_loc = [None, None]
        local_best = [10000., 10000., 10000.,]
        local_best_loc = [None, None, None]
        speed = []
        knobs_num = len(self._knobs)
        default_values = np.random.rand(knobs_num)
        # for i, k in enumerate(list(self._knobs.keys())):
        #     default_values[i] = (self._knobs[k]['value'] - self._knobs[k]['lower']) \
        #         / (self._knobs[k]['upper'] - self._knobs[k]['lower'])
        sampler = qmc.Sobol(d=knobs_num)
        for i in range(len(local_best)):
            # if i == 0:
            #     self._arg_q.put_nowait((i, default_values))
            # else:    
            self._arg_q.put_nowait((i, sampler.random(1)[0]))
            speed.append(np.random.uniform(-0.1, 0.1, (1, knobs_num))[0])

        # wait for any query finish
        count = self._max_tries
        while True:
            count -= 1
            id, values, stats = self._res_q.get()
            print("accept id {} stats {}".format(id, stats))
            self._records.append((list(values), stats))
            if count <= 0:
                break
            # update global best
            if len(stats) > 0:
                for i, j in enumerate(metrics):
                    if stats[j] < global_best[i]:
                        global_best[i] = stats[j]
                        global_best_loc[i] = values
                # update local best
                if stats['elapsed'] < local_best[id]:
                    local_best[id] = stats['elapsed']
                    local_best_loc[id] = values
            
            if stats['fail'] <= 1:
                speed[id] = np.random.uniform(-0.1, 0.1, (1, knobs_num))[0]
                values = sampler.random(1)[0]
            else:
                # update speed
                speed[id] = speed[id] + 0.5 * np.random.rand() * (local_best_loc[id] - values) \
                + 0.5 * np.random.rand() * (global_best_loc[0] - values)
                values += speed[id]
                values = np.clip(values, 0, 1)
            self._arg_q.put_nowait((id, values))

    def consume(self):
        while True:
            time.sleep(0.01)
            try:
                id, values = self._arg_q.get(timeout=3)
                # knobs to settings
                settings = {}
                for i, k in enumerate(self._knobs.keys()):
                    if self._knobs[k]['type'] == 'int/continuous':
                        settings[k] = \
                        round(self._knobs[k]['lower'] + \
                        values[i] * (self._knobs[k]['upper'] - self._knobs[k]['lower']))
                    else:
                        idx = math.floor(values[i] * len(self._knobs[k]['value']))
                        if idx == len(self._knobs[k]['value']):
                            idx -=1
                        settings[k] = self._knobs[k]['value'][idx]
                stats = exec_query(self._sql, self._db, settings)
                self._res_q.put((id, values, stats))
            except Exception as e:
                break

    def run(self):
        self.consume() if self._is_consumer else self.produce()


class PSOSampler(object):

    def __init__(self, sqls, db, knobs, fname='data/sample'):
        self.sqls = sqls
        self.db = db
        self.knobs = knobs
        self.knobs_num = len(knobs)
        self.fname = fname
    
    def sample_on_sql(self, sql: str, max_threads=3):
        q1 = queue.Queue()
        q2 = queue.Queue()
        t = []
        records = []
        exec_query(sql, self.db)
        try:
            analyze_plan = json.dumps(get_query_plan(sql, self.db, analyze=True))
        except Exception as e:
            analyze_plan = {}
        general_plan = json.dumps(get_query_plan(sql, self.db, analyze=False))
        for i in range(max_threads):
            sampler = SinglePSOSampler(sql, self.db, knobs, 100, q1, q2, True, records)
            sampler.start()
            t.append(sampler)
        sampler = SinglePSOSampler(sql, self.db, knobs, 100, q1, q2, False, records)
        sampler.start()
        t.append(sampler)
        for i in range(len(t)):
            try:
                t[i].join()
            except Exception as e:
                break
        with open(self.fname, 'a+') as f:
            for values, stats in records:
                f.write(analyze_plan + "\t")
                f.write(general_plan + "\t")
                f.write(json.dumps(values) + "\t")
                f.write(json.dumps(stats) + "\n")

    def run_samples(self, sqls: list, max_threads=3):
        for i, sql in enumerate(sqls):
            print("sample on index %d sql %s" % (i, sql))
            self.sample_on_sql(sql, max_threads)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--knob_file', '-k')
    parser.add_argument('--db', '-db')
    parser.add_argument('--sqls', '-s')
    parser.add_argument('--output', '-o')
    parser.add_argument('--threads', '-t', type=int, default=1)
    args = parser.parse_args()

    print("start collecting warm-start samples.")

    f = open(args.knob_file)
    knobs = json.load(f)
    f.close()

    sampler = PSOSampler([], args.db, knobs, args.output)
    with open(args.sqls) as f:
        lines = f.read().splitlines()
    sampler.run_samples(lines, args.threads)
