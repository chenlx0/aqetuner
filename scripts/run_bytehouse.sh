nohup ./programs/tso_server --config-file ./ByteHouse_conf/ByteHouse-tso.xml > /dev/null 2>&1 &
nohup ./programs/resource_manager --config-file ./ByteHouse_conf/ByteHouse-resource-manager.xml > /dev/null 2>&1 &
nohup ./programs/clickhouse-server -C ./ByteHouse_conf/ByteHouse-server.xml > /dev/null 2>&1 &
nohup ./programs/clickhouse-server -C ./ByteHouse_conf/ByteHouse-worker1.xml > /dev/null 2>&1 &
nohup ./programs/clickhouse-server -C ./ByteHouse_conf/ByteHouse-worker2.xml > /dev/null 2>&1 &
nohup ./programs/clickhouse-server -C ./ByteHouse_conf/ByteHouse-worker-writer.xml > /dev/null 2>&1 &
nohup ./programs/daemon_manager --config-file ./ByteHouse_conf/ByteHouse-daemon-manager.xml > /dev/null 2>&1 &