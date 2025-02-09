#!/bin/bash

# 获取当前数据库名称
CURRENT_DB=$(~/bvc/ce-2.6/bin/clickhouse client --host=localhost --port=3720 --enable_optimizer=1 --database tpch_100 --multiquery --query "SELECT currentDatabase()" 2>/dev/null)

# 获取所有表名
TABLES=$(~/bvc/ce-2.6/bin/clickhouse client --host=localhost --port=3720 --enable_optimizer=1 --database tpch_100 --multiquery --query "SHOW TABLES" 2>/dev/null)

for TABLE in $TABLES
do
    # 获取表的所有列名
    COLUMNS=$(~/bvc/ce-2.6/bin/clickhouse client --host=localhost --port=3720 --enable_optimizer=1 --database tpch_100 --multiquery --query "DESCRIBE TABLE $TABLE" 2>/dev/null | awk '{print $1}')

    for COLUMN in $COLUMNS
    do
        # 获取列的最大值
        MAX_VALUE=$(~/bvc/ce-2.6/bin/clickhouse client --host=localhost --port=3720 --enable_optimizer=1 --database tpch_100 --multiquery --query "SELECT toUInt64(toDateTime(max($COLUMN))) FROM $TABLE" 2>/dev/null)
        # 获取列的最小值
        MIN_VALUE=$(~/bvc/ce-2.6/bin/clickhouse client --host=localhost --port=3720 --enable_optimizer=1 --database tpch_100 --multiquery --query "SELECT toUInt64(toDateTime(min($COLUMN))) FROM $TABLE" 2>/dev/null)

        echo "$TABLE $COLUMN $MIN_VALUE $MAX_VALUE"
    done
done
