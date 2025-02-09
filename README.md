# Towards Safe Query-level Tuning for ByteHouse

How to generate warm-start samples:

```
python pso.py --knob_file knobs.json --db imdb --sqls workloads/job --output data/job_samples --threads 3
```

the `--threads` control the parallism level of pso sampling.

Initialize Surrogate Model with warm-start samples:

```
python train.py
```


Start Tuning:

```
python tune.py
```
