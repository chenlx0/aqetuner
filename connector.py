import requests
import json
import time
import os

# HOST = 'http://localhost:8123'
HOST = os.getenv('DB_HOST')


def exec_query(sql: str, db: str, settings={}):
    if sql.endswith(';'):
        sql = sql[:-1]
    sql += " FORMAT JSON"
    settings['database'] = db
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
    r = requests.post(HOST, data=sql, params=p)
    r.encoding = 'utf-8'
    content = r.text
    content = content.replace('\\n', '')
    content = content.replace('\\', ' ')
    return json.loads(content)

if __name__ == "__main__":
    print(get_query_plan("select 1", "tpcds", analyze=True))
