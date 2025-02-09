import requests
import json
import time

HOST = 'http://localhost:6725'

DEFAULT_CONFIG = {
        "enable_optimizer": 1,
        "enable_windows_parallel": 1,
        "send_timeout": 400000,
        "plan_optimizer_timeout": 60000,
        "exchange_timeout_ms": 500000,
        "iterative_optimizer_timeout": 30000000,
        "receive_timeout": 400000,
        "cascades_optimizer_timeout": 1000000,
        "max_memory_usage": 500000000000,
        "max_execution_time": 150,
        "use_uncompressed_cache": 1,
        "merge_tree_max_rows_to_use_cache": 280000000000,
        "merge_tree_max_bytes_to_use_cache": 400000000000,
        "distributed_aggregation_memory_efficient": 0,
        # "dialect_type": "ANSI"
}

def exec_query(sql: str, db: str, settings={}):
    if sql.endswith(';'):
        sql = sql[:-1]
    sql += " FORMAT JSON"
    settings['database'] = db
    settings.update(DEFAULT_CONFIG)
    start = time.time()
    try:
        r = requests.post(HOST, data=sql, params=settings)
        if r.status_code != 200:
            print(r.text)
            return {'elapsed':time.time()-start, "fail": 1}
        r.encoding = 'utf-8'
        resp = r.json()
    except Exception as e:
        print(e)
        return {'elapsed':time.time()-start, "fail": 1}
    resp['statistics']['fail'] = 0
    return resp['statistics']

def get_query_plan(sql: str, db: str, analyze=False):
    if sql.endswith(';'):
        sql = sql[:-1]
    prefix = "explain analyze json=1 " if analyze else "explain json=1 "
    sql = prefix + sql
    p = {'database': db, 'enable_optimizer': 1}
    p.update(DEFAULT_CONFIG)
    r = requests.post(HOST, data=sql, params=p)
    r.encoding = 'utf-8'
    content = r.text
    content = content.replace('\\n', '')
    content = content.replace('\\', ' ')
    return json.loads(content)

if __name__ == "__main__":
    print(get_query_plan("select 1", "imdb", analyze=True))
