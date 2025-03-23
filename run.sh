export DB=tpcds
export DB_HOST=http://localhost:8123

mkdir -p checkpoint
mkdir -p data

python pso.py --knob_file tuned_knobs.json --db tpcds --sqls workloads/tpcds --output data/tpcds_samples --threads 1

python train.py

python tune.py
